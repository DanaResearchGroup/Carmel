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
import urllib.error
import urllib.request
from enum import StrEnum

from carmel.config import AgentProvider
from carmel.logger import get_logger

logger = get_logger("agents.model_catalog")

__all__ = [
    "AUTO_PREFIX",
    "AutoFamily",
    "auto_model_name",
    "is_auto_model_name",
    "rank_family_candidates",
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
}

_MAX_LADDER = 4
_CATALOGUE_TIMEOUT_S = 30.0

#: Process-lifetime cache keyed by provider. The catalogue changes on the order of weeks,
#: while a single campaign builds several agents, so re-listing per agent would add
#: latency and a failure mode for no benefit. NOT keyed by api key: the value cached is a
#: list of public model names, and the key never enters it.
_catalogue_cache: dict[AgentProvider, tuple[str, ...]] = {}


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
    if url is None:
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
        ValueError: If ``model_name`` names a family that does not exist, or requests
            ``auto:`` resolution for a provider with no known catalogue endpoint. Both
            are configuration errors, and both fail loudly rather than resolving to some
            arbitrary model the operator did not ask for.
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


def clear_catalogue_cache() -> None:
    """Drop the cached provider catalogues. For tests and long-lived processes."""
    _catalogue_cache.clear()
