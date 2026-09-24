"""Resolve a model FAMILY to the newest concrete model the provider actually serves.

Carmel used to pin exact model names (``gemini-3.5-flash``, ``gemini-pro-latest``) in
:data:`carmel.config.DEFAULT_TIER_MODELS`. Both failure modes of that approach were
observed live on the same afternoon:

- a **dated pin rots**. ``gemini-2.5-flash`` answers ``404 NOT_FOUND: "no longer
  available to new users"`` -- and, importantly, it is still LISTED by the models
  endpoint, so merely checking the catalogue is not enough to notice.
- a **moving alias drifts silently**. ``gemini-pro-latest`` never fails, but it is the
  one name ``genai_prices`` cannot price, so it fell through to a hand-written rate that
  had gone stale by a factor of two against the family it actually aliases. A pin that
  breaks is loud; an alias that changes underneath the budget ledger is not.

So a tier names a FAMILY (``auto:gemini-flash``, ``auto:gemini-pro``) and this module
resolves it, at build time, to the highest-versioned concrete model the provider lists
-- and hands back the rest of the ladder in descending order so a caller can walk down
it when the top choice is temporarily unavailable. That is not hypothetical either:
``gemini-3.1-pro-preview`` returned ``503 UNAVAILABLE`` ("high demand ... usually
temporary") and served normally ninety seconds later.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from carmel.config import AgentProvider
from carmel.logger import get_logger

logger = get_logger("agents.model_catalog")

__all__ = [
    "AUTO_PREFIX",
    "AutoFamily",
    "OpenRouterModelInfo",
    "auto_model_name",
    "fetch_openrouter_catalogue",
    "free_openrouter_model_ids",
    "is_auto_model_name",
    "parse_openrouter_catalogue",
    "rank_family_candidates",
    "rank_free_nemotron_candidates",
    "resolve_model_ladder",
]

#: Marks a Carmel-resolved FAMILY rather than a provider model id. Deliberately not a
#: bare suffix like ``-latest``: providers ship their own ``-latest`` aliases
#: (``gemini-flash-latest``), and a sentinel that could be mistaken for one of those --
#: or accidentally sent to the provider verbatim -- would reintroduce exactly the silent
#: alias-drift this module exists to remove.
AUTO_PREFIX = "auto:"


class AutoFamily(StrEnum):
    """Model families Carmel can resolve to a newest-available member."""

    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    NEMOTRON_FREE = "nemotron-free"


#: The provider whose catalogue each family is resolved against. A family is a set of
#: provider-specific ids, so resolving it against another provider's catalogue would
#: either match nothing or fall back to a static ladder of ids that provider never serves.
_FAMILY_PROVIDERS: dict[AutoFamily, AgentProvider] = {
    AutoFamily.GEMINI_FLASH: AgentProvider.GOOGLE,
    AutoFamily.GEMINI_PRO: AgentProvider.GOOGLE,
    AutoFamily.NEMOTRON_FREE: AgentProvider.OPENROUTER,
}


def auto_model_name(family: AutoFamily) -> str:
    """Return the sentinel model name for ``family`` (e.g. ``"auto:gemini-flash"``)."""
    return f"{AUTO_PREFIX}{family.value}"


def is_auto_model_name(model_name: str) -> bool:
    """Return True if ``model_name`` is a Carmel family sentinel, not a provider id."""
    return model_name.startswith(AUTO_PREFIX)


# Match ONLY plain text-generation members of each family. The exclusions are the whole
# point: the provider's catalogue mixes capability variants into the same name space, and
# quietly resolving a tier to one of them would be worse than a stale pin. Ruled out by
# construction, because none of them can match the anchored pattern:
#   -lite            a cheaper, weaker sibling -- a real choice, never an automatic one
#   -image / -tts    different modality; would not answer a structured-output call
#   -customtools     a tool-calling variant with its own contract
#   -001             a provider-side dated pin of an older release
#   gemini-*-latest  the provider's own moving alias, i.e. the thing we are replacing
#   lyria- / deep-research-  other product lines that merely contain "pro"
#
# The optional minor version accommodates both spellings the provider uses in practice
# (``gemini-3-pro-preview`` and ``gemini-3.1-pro-preview``); a missing minor sorts as 0.
_FAMILY_PATTERNS: dict[AutoFamily, re.Pattern[str]] = {
    AutoFamily.GEMINI_FLASH: re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash(?:-preview)?$"),
    AutoFamily.GEMINI_PRO: re.compile(r"^gemini-(\d+)(?:\.(\d+))?-pro(?:-preview)?$"),
}

#: Last-resort ladders used only when the provider's catalogue cannot be read at all
#: (network down, key rejected). Verified served on 2026-07-28. These will rot -- that is
#: accepted and bounded: they apply only when discovery has already failed, and a rotten
#: entry then surfaces as a loud provider error rather than as a wrong-but-plausible
#: choice. Discovery, not this table, is the mechanism that is meant to keep working.
_STATIC_FALLBACK_LADDERS: dict[AutoFamily, tuple[str, ...]] = {
    AutoFamily.GEMINI_FLASH: ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview"),
    AutoFamily.GEMINI_PRO: ("gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro"),
}

#: Provider catalogue endpoints. Only providers listed here support ``auto:`` resolution;
#: anything else must name a concrete model, and says so rather than guessing.
_CATALOGUE_URLS: dict[AgentProvider, str] = {
    AgentProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
    AgentProvider.OPENROUTER: "https://openrouter.ai/api/v1/models",
}

#: OpenRouter marks its zero-cost variants with this id suffix.
OPENROUTER_FREE_SUFFIX = ":free"

#: Context window the OpenRouter DEV default is chosen for. Candidates at or above it are
#: preferred; if none reaches it, the largest-context free Nemotron is used instead.
_NEMOTRON_TARGET_CONTEXT = 1_000_000

_MAX_LADDER = 4
_CATALOGUE_TIMEOUT_S = 30.0

#: How long a read of OpenRouter's catalogue is trusted. Free status is a PRICE, and a
#: price can change mid-process; a model that starts charging must stop counting as
#: free within this bound rather than for the life of the process.
_OPENROUTER_CATALOGUE_TTL_S = 600.0

#: Process-lifetime cache keyed by provider. The catalogue changes on the order of weeks,
#: while a single campaign builds several agents, so re-listing per agent would add
#: latency and a failure mode for no benefit. NOT keyed by api key: the value cached is a
#: list of public model names, and the key never enters it.
_catalogue_cache: dict[AgentProvider, tuple[str, ...]] = {}


@dataclass(frozen=True)
class OpenRouterModelInfo:
    """One entry of OpenRouter's public model catalogue, reduced to what Carmel uses.

    Attributes:
        model_id: The OpenRouter model id (e.g. ``"vendor/model:free"``).
        context_length: Advertised context window in tokens, or None if not reported.
        zero_priced: True only when both the prompt and completion prices are the
            literal string ``"0"``. A missing or unparseable price is NOT zero.
    """

    model_id: str
    context_length: int | None
    zero_priced: bool

    @property
    def is_free(self) -> bool:
        """Return True if the catalogue prices this model at zero AND its id says ``:free``.

        Both are required: the suffix alone is only a naming convention, and a zero price
        alone could be a promotional rate on a paid id that silently ends.
        """
        return self.zero_priced and self.model_id.endswith(OPENROUTER_FREE_SUFFIX)


#: Cache of OpenRouter's catalogue as ``(catalogue, read_at)``, trusted for
#: ``_OPENROUTER_CATALOGUE_TTL_S``. Only a SUCCESSFUL read is cached, so a network blip
#: does not pin "nothing is free"; an EXPIRED entry is never reused, even when the
#: refetch fails, so a stale "free" can never outlive the TTL.
_openrouter_catalogue_cache: list[tuple[tuple[OpenRouterModelInfo, ...], float]] = []


def _clock() -> float:
    """Return the monotonic time the catalogue TTL is measured on; patched in tests."""
    return time.monotonic()


def _version_key(match: re.Match[str]) -> tuple[int, int]:
    """Return ``(major, minor)`` for a family-pattern match; absent minor sorts as 0."""
    return (int(match.group(1)), int(match.group(2) or 0))


def rank_family_candidates(model_names: object, family: AutoFamily) -> list[str]:
    """Return ``family`` members of ``model_names``, newest version first.

    Pure and offline so the selection rule can be tested against a recorded catalogue
    without a network call -- the ranking is the part that decides what Carmel spends
    money on, and it must be verifiable independently of whatever the provider happens
    to be serving on the day the tests run.

    Args:
        model_names: An iterable of provider model ids.
        family: The family to select from.

    Returns:
        Matching model ids sorted by descending ``(major, minor)`` version. Ties are
        broken by name so the result is deterministic. Empty if nothing matches.
    """
    pattern = _FAMILY_PATTERNS[family]
    matched: list[tuple[tuple[int, int], str]] = []
    for name in model_names:  # type: ignore[attr-defined]
        match = pattern.match(name)
        if match is not None:
            matched.append((_version_key(match), name))
    matched.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    return [name for _, name in matched]


def parse_openrouter_catalogue(payload: object) -> tuple[OpenRouterModelInfo, ...]:
    """Parse the JSON body of ``GET /api/v1/models`` into catalogue entries. Never raises.

    Pure and offline, like :func:`rank_family_candidates`: whether a model counts as free
    decides whether a DEV run may call it, so the rule is tested against recorded payloads.
    Malformed entries are skipped rather than guessed at.

    Args:
        payload: The decoded JSON response.

    Returns:
        One entry per well-formed catalogue item; empty for a malformed payload.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return ()
    entries: list[OpenRouterModelInfo] = []
    for item in payload["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            continue
        pricing = item.get("pricing")
        zero_priced = isinstance(pricing, dict) and pricing.get("prompt") == "0" and pricing.get("completion") == "0"
        context = item.get("context_length")
        context_length = context if isinstance(context, int) and not isinstance(context, bool) else None
        entries.append(OpenRouterModelInfo(item["id"], context_length, zero_priced))
    return tuple(entries)


def free_openrouter_model_ids(catalogue: Iterable[OpenRouterModelInfo]) -> frozenset[str]:
    """Return the ids of every model in ``catalogue`` that is free by both tests."""
    return frozenset(entry.model_id for entry in catalogue if entry.is_free)


def rank_free_nemotron_candidates(catalogue: Iterable[OpenRouterModelInfo]) -> list[str]:
    """Return free NVIDIA Nemotron ids, largest context window first.

    Candidates reaching :data:`_NEMOTRON_TARGET_CONTEXT` sort ahead of the rest by
    construction, so when none does the head of the list is simply the largest free
    Nemotron. Ties are broken by id so the result is deterministic.
    """
    candidates = [
        entry
        for entry in catalogue
        if entry.is_free and entry.model_id.startswith("nvidia/") and "nemotron" in entry.model_id.lower()
    ]
    candidates.sort(key=lambda entry: (entry.context_length or 0, entry.model_id), reverse=True)
    return [entry.model_id for entry in candidates]


def fetch_openrouter_catalogue() -> tuple[OpenRouterModelInfo, ...]:
    """Read OpenRouter's public model catalogue, cached for a bounded TTL. Never raises.

    The endpoint needs no key, so none is sent. Returns an empty tuple on any failure;
    unlike the Gemini path there is no static fallback, because a model that cannot be
    shown to be free must be refused rather than assumed free. For the same reason a
    failed refetch after the TTL fails CLOSED: the expired entry is dropped, not reused.

    Same consent and budget reasoning as :func:`_fetch_catalogue`: reached only from
    ``build_model`` after its consent check, against a fixed URL, with a small
    provider-bounded body.
    """
    now = _clock()
    if _openrouter_catalogue_cache:
        catalogue, read_at = _openrouter_catalogue_cache[0]
        if now - read_at < _OPENROUTER_CATALOGUE_TTL_S:
            return catalogue
        _openrouter_catalogue_cache.clear()
    request = urllib.request.Request(_CATALOGUE_URLS[AgentProvider.OPENROUTER])
    try:
        with urllib.request.urlopen(request, timeout=_CATALOGUE_TIMEOUT_S) as response:  # noqa: S310 - fixed https URL
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        logger.warning("could not read the openrouter model catalogue (%s)", type(exc).__name__)
        return ()
    catalogue = parse_openrouter_catalogue(payload)
    _openrouter_catalogue_cache.append((catalogue, now))
    return catalogue


def _fetch_catalogue(provider: AgentProvider, api_key: str) -> tuple[str, ...]:
    """List model ids the provider will generate content with. Never raises.

    Returns an empty tuple on any failure, which callers translate into "use the static
    fallback ladder" -- a catalogue lookup failing must degrade the CHOICE of model, not
    break the run.

    Consent and budget, documented: this is the one HTTP egress point in the agents
    stack that deliberately does NOT go through ``BudgetLedger`` (unlike
    ``HttpFetchTool``, whose every call reserves and settles against it) and has no
    size cap of its own. That is a considered choice, not an oversight:
      * Consent IS already enforced -- this function is only ever reached via
        ``resolve_model_ladder`` from ``build_model``, which raises before this point
        if ``config.external_provider_consent`` is False (see the check immediately
        above the ``resolve_model_ladder`` call in ``carmel/agents/models.py``). There
        is no path to this function that bypasses that gate.
      * The URL is a fixed, hardcoded entry from ``_CATALOGUE_URLS`` -- never
        attacker- or LLM-influenced -- so the SSRF threat model that ``HttpFetchTool``
        guards against (an adversary choosing an arbitrary destination) does not apply
        here.
      * The response body is a small, provider-controlled JSON model listing (not
        arbitrary attacker content), so the decompression-bomb-scale risk
        ``HttpFetchTool``'s byte cap defends against is not in play either; it is read
        whole via ``response.read()`` rather than streamed.
    If this function is ever changed to accept a caller-supplied URL, or to fetch
    something whose size is not provider-bounded, it must gain the same ledger
    reservation and streaming cap ``HttpFetchTool`` uses -- at that point the
    reasoning above no longer holds.
    """
    url = _CATALOGUE_URLS.get(provider)
    if url is None or provider == AgentProvider.OPENROUTER:
        return ()

    request = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=_CATALOGUE_TIMEOUT_S) as response:  # noqa: S310 - fixed https URL
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        # Deliberately does not log `exc` at error level with a traceback: a failed
        # catalogue read is an expected, recoverable condition, not a defect.
        logger.warning(
            "could not read the %s model catalogue (%s); falling back to the static model ladder",
            provider.value,
            type(exc).__name__,
        )
        return ()

    names: list[str] = []
    for entry in payload.get("models", []):
        if "generateContent" not in entry.get("supportedGenerationMethods", []):
            continue
        name = entry.get("name", "")
        if name:
            names.append(name.removeprefix("models/"))
    return tuple(names)


def resolve_model_ladder(model_name: str, provider: AgentProvider, api_key: str) -> list[str]:
    """Resolve a model name to an ordered list of models to try, best first.

    A concrete model id resolves to itself alone: naming an exact model is a deliberate
    instruction, and silently substituting a different one would be a worse surprise than
    any outage.

    Args:
        model_name: Either a concrete provider model id, or an ``auto:<family>`` sentinel.
        provider: The provider whose catalogue to consult.
        api_key: The key used to read the catalogue. Never logged or cached.

    Returns:
        Model ids in preference order, newest first, capped at a handful of fallbacks.

    Raises:
        ValueError: If ``model_name`` names a family that does not exist, requests
            ``auto:`` resolution for a provider with no known catalogue endpoint or for a
            family that provider does not serve, or names the OpenRouter free family
            while its catalogue lists no free Nemotron. All fail loudly rather than
            resolving to some arbitrary model the operator did not ask for.
    """
    if not is_auto_model_name(model_name):
        return [model_name]

    family_value = model_name.removeprefix(AUTO_PREFIX)
    try:
        family = AutoFamily(family_value)
    except ValueError as exc:
        known = ", ".join(auto_model_name(f) for f in AutoFamily)
        raise ValueError(f"unknown model family {model_name!r}; known families: {known}") from exc

    if provider not in _CATALOGUE_URLS:
        raise ValueError(
            f"provider {provider.value!r} does not support {AUTO_PREFIX!r} model resolution; "
            "set agents.model_name to a concrete model id for this provider"
        )

    if _FAMILY_PROVIDERS[family] != provider:
        raise ValueError(
            f"model family {model_name!r} is served by {_FAMILY_PROVIDERS[family].value!r}, not "
            f"{provider.value!r}; set agents.model_name to a concrete model id for this provider"
        )

    if family == AutoFamily.NEMOTRON_FREE:
        return _resolve_free_nemotron(model_name)

    if provider not in _catalogue_cache:
        _catalogue_cache[provider] = _fetch_catalogue(provider, api_key)

    ladder = rank_family_candidates(_catalogue_cache[provider], family)
    if not ladder:
        ladder = list(_STATIC_FALLBACK_LADDERS[family])
        logger.warning(
            "no %s models found in the %s catalogue; falling back to the static ladder %r",
            family.value,
            provider.value,
            ladder,
        )
    else:
        logger.info("resolved %s to %r (fallbacks: %r)", model_name, ladder[0], ladder[1:_MAX_LADDER])

    return ladder[:_MAX_LADDER]


def _resolve_free_nemotron(model_name: str) -> list[str]:
    """Resolve the OpenRouter free-Nemotron family against the live catalogue.

    Raises:
        ValueError: If the catalogue is unreadable or lists no free Nemotron. There is
            deliberately no static fallback: a pinned id could not be shown to be free.
    """
    ladder = rank_free_nemotron_candidates(fetch_openrouter_catalogue())
    if not ladder:
        raise ValueError(
            f"cannot resolve {model_name!r}: the OpenRouter catalogue is unreachable or lists no "
            "free Nemotron model; set agents.model_name to a concrete ':free' model id"
        )
    logger.info("resolved %s to %r (fallbacks: %r)", model_name, ladder[0], ladder[1:_MAX_LADDER])
    return ladder[:_MAX_LADDER]


def clear_catalogue_cache() -> None:
    """Drop the cached provider catalogues. For tests and long-lived processes."""
    _catalogue_cache.clear()
    _openrouter_catalogue_cache.clear()
