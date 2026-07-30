"""Scholarly index adapters: OpenAlex/Crossref search, and open-access resolution.

The search backends exist because the generic
:class:`~carmel.agents.tools.search.HttpSearchTool` assumes an operator-supplied
endpoint plus a bearer key, which no scholarly index requires — and demanding one made
them impossible to configure at all. The second half of the module is
:class:`OpenAccessResolver`: deterministic DOI/title -> open-access-PDF resolution over
a pluggable list of OA indexes (OpenAlex, Unpaywall, Crossref TDM links, Semantic
Scholar, CORE, DOAJ, ChemRxiv, arXiv). ChemRxiv is registered but disabled by default;
see :data:`DEFAULT_ENABLED_PROVIDERS`. FatCat / Internet Archive Scholar was removed
entirely on 2026-07-29 -- see the note by the provider registry.

Both adapters are shaped by a live probe of 60 combustion-kinetics works rather than by
the APIs' own documentation, because the two disagree sharply:

- **An OA flag is not a fetchable paper.** 29/60 works (48%) were flagged
  ``is_oa``, but only 11 (18%) carried any full-text URL, and only 2 (3.3%) yielded
  text the extractor could actually read. Callers must treat every result as
  *possibly* acquirable and route the rest to manual acquisition.
- **An advertised PDF URL is frequently not a PDF.** 5 of those 11 served an HTML
  landing page. :attr:`SearchResult.pdf_url` is therefore documented as advertised-only;
  verification belongs to whoever fetches the bytes.
- **Repository beats publisher.** Publisher-hosted links returned HTTP 403 to
  non-browser clients in the earlier probe (0/6 usable); preferring
  ``host_type == "repository"`` is what lifted the success rate off zero. Hence
  :func:`_best_location`.

Neither adapter spoofs a browser User-Agent to defeat publisher bot-blocks: a 403 is
treated as "ask a human for this paper", not as an obstacle to route around.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol
from urllib.parse import quote, quote_plus
from xml.etree import ElementTree

from carmel.agents.budget import BudgetLedger
from carmel.agents.tools.search import (
    SearchError,
    SearchResult,
    budgeted_get_json,
    budgeted_get_raw,
    default_opener,
)
from carmel.logger import get_logger

logger = get_logger("agents.tools.academic")

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2"
SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper"
CORE_ENDPOINT = "https://api.core.ac.uk/v3/search/works"
DOAJ_ENDPOINT = "https://doaj.org/api/search/articles"
CHEMRXIV_ENDPOINT = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"

#: Sent on every request so the upstream operator can identify (and contact us about)
#: this client rather than silently rate-limiting it.
USER_AGENT = "Carmel/0.1 (+https://github.com/DanaResearchGroup/Carmel)"

#: Per-request timeout for OA *index* lookups -- the small JSON/Atom metadata calls
#: that ``_lookup_*`` methods make to discover whether a paper has an OA copy, not the
#: (much larger, and much slower) actual PDF/document fetch, which is a different code
#: path in a different tool and keeps its own ~30s timeout. A dead index (FatCat's
#: api.fatcat.wiki timed out at the default 30s on every one of 12 probed DOIs, costing
#: ~6 minutes of wall-clock on a single 12-paper run) should cost at most a few seconds
#: per paper, not tens of seconds, and a short timeout here can never truncate a real
#: document download because it is never used for one.
OA_INDEX_LOOKUP_TIMEOUT_S = 10.0

#: Host types worth trying to fetch directly, best first. Publisher-hosted copies are
#: last because they bot-block plain HTTP clients.
_HOST_TYPE_PREFERENCE = ("repository", "publisher")

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)

#: Crossref ``type`` values that are never a readable paper. A live probe for
#: "ignition delay time shock tube methane" returned two of eight results as
#: ``component`` entries -- the ``.s001``/``.s002`` supplementary-information files of a
#: single article, carrying that article's full title. They survive DOI deduplication
#: (their DOIs genuinely differ) and would otherwise be handed to the agent as two extra
#: candidate "papers", spending fetch budget and manual-acquisition attention on
#: material that is not the paper. This is a DENYLIST rather than an allowlist so that a
#: newly-minted legitimate type is never silently dropped.
_CROSSREF_NON_PAPER_TYPES = frozenset(
    {
        "component",
        "peer-review",
        "grant",
        "journal-issue",
        "journal-volume",
        "journal",
        "book-series",
        "report-series",
    }
)


def normalize_doi(raw: Any) -> str | None:
    """Reduce any DOI spelling to a bare, lowercased ``10.xxxx/yyy``.

    OpenAlex returns ``https://doi.org/10.1016/j.combustflame...`` while Crossref
    returns the bare form; normalizing both makes cross-backend deduplication and
    acquisition-manifest matching possible at all.

    Args:
        raw: A DOI in any of the common spellings, or any non-string value.

    Returns:
        The bare lowercased DOI, or ``None`` if ``raw`` is not a usable DOI.
    """
    if not isinstance(raw, str):
        return None
    candidate = _DOI_PREFIX_RE.sub("", raw.strip()).strip().lower()
    return candidate if candidate.startswith("10.") else None


def _abstract_from_inverted_index(index: Any, *, max_chars: int = 600) -> str:
    """Rebuild readable text from OpenAlex's ``abstract_inverted_index``.

    OpenAlex ships abstracts as ``{word: [positions]}`` rather than as prose. Without
    this the agent would choose papers from titles alone, which is materially worse at
    telling a review apart from a primary measurement.

    Args:
        index: The raw ``abstract_inverted_index`` value, of unknown shape.
        max_chars: Truncation ceiling; abstracts are advisory context, not evidence.

    Returns:
        The reconstructed abstract, or ``""`` when the shape is unusable.
    """
    if not isinstance(index, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, word))
    if not positioned:
        return ""
    positioned.sort()
    text = " ".join(word for _, word in positioned)
    return text[:max_chars]


def _best_location(locations: Any) -> tuple[str | None, str]:
    """Pick the most fetchable full-text location from OpenAlex ``locations[]``.

    Args:
        locations: The raw ``locations`` list, of unknown shape.

    Returns:
        ``(pdf_url, repository_display_name)``; ``(None, "")`` when nothing carries a
        full-text URL. Repository-hosted copies win over publisher-hosted ones, and a
        pdf_url on a host typed neither of those (e.g. an OpenAlex source of type
        ``journal``) is kept as a LAST RESORT rather than dropped: this function used
        to discard exactly those, which is one of the two ways a genuinely fetchable
        OA PDF ended up shown to the agent as "FULL TEXT: no" (the other being
        ``best_oa_location``, a distinct top-level field this list never contains --
        see the fallback in :meth:`OpenAlexSearchTool.search`).
    """
    if not isinstance(locations, list):
        return None, ""

    best_rank = len(_HOST_TYPE_PREFERENCE) + 1
    best: tuple[str | None, str] = (None, "")
    for location in locations:
        if not isinstance(location, dict):
            continue
        pdf_url = location.get("pdf_url")
        if not isinstance(pdf_url, str) or not pdf_url:
            continue
        source = location.get("source")
        source = source if isinstance(source, dict) else {}
        host_type = location.get("host_type") or source.get("type") or ""
        rank = (
            _HOST_TYPE_PREFERENCE.index(host_type) if host_type in _HOST_TYPE_PREFERENCE else len(_HOST_TYPE_PREFERENCE)
        )
        if rank < best_rank:
            name = source.get("display_name")
            best_rank = rank
            best = (pdf_url, name if isinstance(name, str) else "")
    return best


def _best_oa_location_pdf(best_oa_location: Any) -> tuple[str | None, str]:
    """Extract ``(pdf_url, source_display_name)`` from OpenAlex ``best_oa_location``.

    Args:
        best_oa_location: The raw top-level ``best_oa_location`` value, of unknown
            shape (it is ``None`` for non-OA works).

    Returns:
        ``(pdf_url, display_name)``, or ``(None, "")`` when there is no usable PDF URL.
    """
    if not isinstance(best_oa_location, dict):
        return None, ""
    pdf_url = best_oa_location.get("pdf_url")
    if not isinstance(pdf_url, str) or not pdf_url:
        return None, ""
    source = best_oa_location.get("source")
    source = source if isinstance(source, dict) else {}
    name = source.get("display_name")
    return pdf_url, name if isinstance(name, str) else ""


def dedupe_by_doi(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Drop repeat papers, preserving order and preferring entries with a full text.

    The live probe returned the same paper ("Modeling nitrogen chemistry in
    combustion") twice as two separate repository records. Left in, each duplicate
    spends a second fetch against a hard budget for bytes already held.

    Args:
        results: Search results, possibly containing duplicates.

    Returns:
        Deduplicated results. Entries without a DOI are never merged (there is nothing
        reliable to merge them on) but are still deduplicated on exact URL.
    """
    by_key: dict[str, int] = {}
    kept: list[SearchResult] = []
    for result in results:
        key = f"doi:{result.doi}" if result.doi else f"url:{result.url}"
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = len(kept)
            kept.append(result)
            continue
        # Same paper seen twice: keep whichever copy actually offers full text.
        if kept[existing].pdf_url is None and result.pdf_url is not None:
            kept[existing] = result
    return kept


class _KeylessSearchTool:
    """Shared plumbing for the keyless scholarly backends."""

    def __init__(
        self,
        *,
        ledger: BudgetLedger,
        external_provider_consent: bool,
        contact_email: str | None = None,
        opener: Callable[..., Any] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Construct the tool.

        Args:
            ledger: Budget ledger; every search call is reserved and settled through it.
            external_provider_consent: Operator opt-in to third-party network egress.
                Required, with no default -- see ``budgeted_get_json``.
            contact_email: Optional address for the API's "polite pool" (higher rate
                limits). Omitted from the URL entirely when not supplied.
            opener: Injected ``(url, headers=..., timeout_s=...) -> response`` opener,
                for tests. Defaults to a real urllib GET.
            timeout_s: Per-request socket timeout.
        """
        self._ledger = ledger
        self._external_provider_consent = external_provider_consent
        self._contact_email = contact_email
        self._opener = opener if opener is not None else default_opener
        self._timeout_s = timeout_s

    def _get(self, url: str) -> Any | None:
        """Perform one budgeted, JSON-parsed GET."""
        return budgeted_get_json(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            ledger=self._ledger,
            opener=self._opener,
            timeout_s=self._timeout_s,
            external_provider_consent=self._external_provider_consent,
        )


class OpenAlexSearchTool(_KeylessSearchTool):
    """Keyless search over OpenAlex, preferring repository-hosted full text."""

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search OpenAlex works.

        Args:
            query: Free-text search query.
            limit: Maximum number of results requested.

        Returns:
            Deduplicated results, never raising on an unexpected payload shape.
        """
        url = f"{OPENALEX_ENDPOINT}?search={quote_plus(query)}&per-page={max(1, limit)}"
        if self._contact_email:
            url = f"{url}&mailto={quote_plus(self._contact_email)}"

        payload = self._get(url)
        works = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(works, list):
            logger.warning("OpenAlex response had no usable results list")
            return []

        results: list[SearchResult] = []
        for work in works:
            if not isinstance(work, dict):
                continue
            title = work.get("title") or work.get("display_name")
            if not isinstance(title, str) or not title:
                continue
            doi = normalize_doi(work.get("doi"))
            open_access = work.get("open_access")
            open_access = open_access if isinstance(open_access, dict) else {}
            pdf_url, repository = _best_location(work.get("locations"))
            if pdf_url is None:
                # ``best_oa_location`` is a TOP-LEVEL field, not an entry of
                # ``locations[]``; ignoring it dropped real publisher OA PDFs (the
                # queued-as-"paywalled" defect's search-side half).
                pdf_url, repository = _best_oa_location_pdf(work.get("best_oa_location"))

            landing = work.get("id")
            fetch_target = pdf_url or (f"https://doi.org/{doi}" if doi else None)
            if fetch_target is None and isinstance(landing, str) and landing:
                fetch_target = landing
            if fetch_target is None:
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=fetch_target,
                    snippet=_abstract_from_inverted_index(work.get("abstract_inverted_index")),
                    source="openalex",
                    doi=doi,
                    pdf_url=pdf_url,
                    is_open_access=bool(open_access.get("is_oa")),
                    repository=repository,
                )
            )
        return dedupe_by_doi(results)[:limit]


class CrossrefSearchTool(_KeylessSearchTool):
    """Keyless search over Crossref.

    Crossref indexes essentially all DOIs, including paywalled ones, so it is the
    better backend for *identifying* a paper the campaign needs; it rarely yields
    fetchable full text. Results from here are expected to feed manual acquisition.
    """

    def search(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        """Search Crossref works.

        Args:
            query: Free-text search query.
            limit: Maximum number of results requested.

        Returns:
            Deduplicated results, never raising on an unexpected payload shape.
        """
        url = f"{CROSSREF_ENDPOINT}?query={quote_plus(query)}&rows={max(1, limit)}"
        if self._contact_email:
            url = f"{url}&mailto={quote_plus(self._contact_email)}"

        payload = self._get(url)
        message = payload.get("message") if isinstance(payload, dict) else None
        items = message.get("items") if isinstance(message, dict) else None
        if not isinstance(items, list):
            logger.warning("Crossref response had no usable items list")
            return []

        results: list[SearchResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") in _CROSSREF_NON_PAPER_TYPES:
                continue
            titles = item.get("title")
            title = titles[0] if isinstance(titles, list) and titles else None
            if not isinstance(title, str) or not title:
                continue
            doi = normalize_doi(item.get("DOI"))
            pdf_url = _crossref_pdf_link(item.get("link"))

            fetch_target = pdf_url or (f"https://doi.org/{doi}" if doi else None)
            if fetch_target is None:
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=fetch_target,
                    snippet="",
                    source="crossref",
                    doi=doi,
                    pdf_url=pdf_url,
                    # Crossref does not assert open-access status; claiming it here
                    # would invent a fact the API never supplied.
                    is_open_access=False,
                    repository="",
                )
            )
        return dedupe_by_doi(results)[:limit]


def _crossref_pdf_link(links: Any) -> str | None:
    """Extract a PDF full-text link from a Crossref ``link[]`` array.

    Args:
        links: The raw ``link`` value, of unknown shape.

    Returns:
        The first ``application/pdf`` URL, or ``None``.
    """
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("content-type") != "application/pdf":
            continue
        url = link.get("URL")
        if isinstance(url, str) and url:
            return url
    return None


# --------------------- deterministic open-access resolution ---------------------
#
# This exists because of a real-run defect: 12 papers were queued for manual
# acquisition as "paywalled" purely on the proposing LLM's say-so, and 5 of the 12
# were in fact open access (2 with a directly fetchable publisher PDF). Whether a
# paper has an OA copy is a question for the OA indexes, keyed on DOI, in plain
# deterministic code -- it must NEVER depend on model judgement.


class OaTier(IntEnum):
    """Preference rank of an open-access copy. Lower is tried first.

    A publisher-hosted OA PDF is (for gold OA) the version of record; repository and
    archive copies are the same accepted text hosted elsewhere; a preprint is NOT the
    version of record at all and must always rank last, clearly labelled.
    """

    PUBLISHER = 0
    REPOSITORY = 1
    PREPRINT = 2


@dataclass(frozen=True)
class OaCandidate:
    """One fetchable open-access PDF candidate, with its preference tier."""

    url: str
    tier: OaTier


#: Hard ceiling on index lookup calls per resolved paper. Eight providers are
#: registered today and each makes at most one call, so this never trips in normal
#: operation; it is the bound that keeps a 12-paper run at <=96 small JSON/Atom
#: lookups no matter how many providers are added later, instead of letting provider
#: growth silently multiply network fan-out. Providers beyond the cap are skipped
#: loudly in the resolution note.
MAX_OA_LOOKUP_CALLS_PER_PAPER = 10

#: Providers actually queried by :class:`OpenAccessResolver` by default. All
#: registered providers are listed in the class's ``all_providers`` tuple; only the
#: ones named here are kept, so disabling a provider without deleting its parser/
#: lookup method is just leaving its name out of this set.
#:
#: ChemRxiv is registered (parser + lookup method preserved -- chemistry preprints
#: are genuinely relevant to this domain) but excluded here as of 2026-07-29: its
#: ``public-api/v1/items?term=`` endpoint returned 403/404 for all 12 probed titles
#: in a live probe, so querying it by default is pure latency/log noise. Re-enable
#: by adding "ChemRxiv" back to this set once the endpoint is re-verified alive.
DEFAULT_ENABLED_PROVIDERS: frozenset[str] = frozenset(
    {
        "OpenAlex",
        "Unpaywall",
        "Crossref",
        "Semantic Scholar",
        "CORE",
        "DOAJ",
        "arXiv",
    }
)

#: Results requested from each search-shaped index (CORE, ChemRxiv, arXiv). Small on
#: purpose: the match criteria are exact (DOI or exact title), so the wanted paper is
#: either in the first few hits or not there at all.
_PROVIDER_SEARCH_LIMIT = 5

_TITLE_JUNK_RE = re.compile(r"[^a-z0-9]+")


def _normalized_title(title: str) -> str:
    """Reduce a title to a punctuation/case/whitespace-insensitive comparison key.

    Used for the EXACT-match requirement of the title-matched providers (ChemRxiv,
    arXiv): "Ammonia Combustion Kinetics: A Study" and "ammonia combustion kinetics --
    a study" are the same title, while a merely similar title must never match --
    fetching the wrong paper is far worse than a miss.

    Args:
        title: The raw title.

    Returns:
        The normalized comparison key.
    """
    return _TITLE_JUNK_RE.sub(" ", title.casefold()).strip()


def openalex_oa_candidates(work: Any) -> list[OaCandidate]:
    """Tiered OA PDF candidates from one OpenAlex work record.

    ``best_oa_location.pdf_url`` comes first (for gold OA that is the publisher's
    version of record, hence :attr:`OaTier.PUBLISHER`), followed by every
    ``locations[].pdf_url`` whose location is explicitly ``is_oa: true`` as
    :attr:`OaTier.REPOSITORY` copies. A location's pdf_url WITHOUT the OA flag is
    excluded: using it would assert open access the index never claimed.

    Args:
        work: The raw OpenAlex work payload, of unknown/untrusted shape.

    Returns:
        Deduplicated candidates in preference order; empty for unusable shapes.
    """
    if not isinstance(work, dict):
        return []
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    best = work.get("best_oa_location")
    if isinstance(best, dict):
        pdf_url = best.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url:
            candidates.append(OaCandidate(url=pdf_url, tier=OaTier.PUBLISHER))
            seen.add(pdf_url)
    locations = work.get("locations")
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict) or location.get("is_oa") is not True:
                continue
            pdf_url = location.get("pdf_url")
            if isinstance(pdf_url, str) and pdf_url and pdf_url not in seen:
                candidates.append(OaCandidate(url=pdf_url, tier=OaTier.REPOSITORY))
                seen.add(pdf_url)
    return candidates


def openalex_oa_pdf_urls(work: Any) -> list[str]:
    """Flat-URL view of :func:`openalex_oa_candidates` (kept for existing callers)."""
    return [candidate.url for candidate in openalex_oa_candidates(work)]


def unpaywall_oa_candidates(payload: Any) -> list[OaCandidate]:
    """Tiered OA PDF candidates from one Unpaywall DOI record.

    ``best_oa_location.url_for_pdf`` first as :attr:`OaTier.PUBLISHER` (it is
    Unpaywall's own version-of-record preference), then every
    ``oa_locations[].url_for_pdf`` as :attr:`OaTier.REPOSITORY`. Everything in
    ``oa_locations`` is open access by construction of the API.

    Args:
        payload: The raw Unpaywall response payload, of unknown/untrusted shape.

    Returns:
        Deduplicated candidates in preference order; empty for unusable shapes.
    """
    if not isinstance(payload, dict):
        return []
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        url_for_pdf = best.get("url_for_pdf")
        if isinstance(url_for_pdf, str) and url_for_pdf:
            candidates.append(OaCandidate(url=url_for_pdf, tier=OaTier.PUBLISHER))
            seen.add(url_for_pdf)
    oa_locations = payload.get("oa_locations")
    if isinstance(oa_locations, list):
        for location in oa_locations:
            if not isinstance(location, dict):
                continue
            url_for_pdf = location.get("url_for_pdf")
            if isinstance(url_for_pdf, str) and url_for_pdf and url_for_pdf not in seen:
                candidates.append(OaCandidate(url=url_for_pdf, tier=OaTier.REPOSITORY))
                seen.add(url_for_pdf)
    return candidates


def unpaywall_oa_pdf_urls(payload: Any) -> list[str]:
    """Flat-URL view of :func:`unpaywall_oa_candidates` (kept for existing callers)."""
    return [candidate.url for candidate in unpaywall_oa_candidates(payload)]


#: Crossref ``link[].intended-application`` values that mark a publisher-sanctioned
#: machine-readable full-text link (text-and-data-mining / plagiarism-check feeds).
#: ``unspecified`` is deliberately absent: those links are frequently the HTML landing
#: page re-advertised as a PDF, and the publisher has sanctioned nothing about them.
_CROSSREF_TDM_INTENDED = frozenset({"text-mining", "similarity-checking"})


def crossref_tdm_candidates(payload: Any) -> list[OaCandidate]:
    """Publisher-sanctioned full-text PDF links from one Crossref work record.

    Crossref never asserts open access, but publishers deposit ``link[]`` entries
    intended for text-mining / similarity-checking; where those are ``application/pdf``
    they are publisher-hosted full text sanctioned for machine retrieval, worth ONE
    fetch attempt. A 401/402/403 outcome is then an honest PAYWALLED observation.

    Args:
        payload: The raw Crossref ``works/{doi}`` response payload, of unknown shape.

    Returns:
        Deduplicated :attr:`OaTier.PUBLISHER` candidates; empty for unusable shapes.
    """
    message = payload.get("message") if isinstance(payload, dict) else None
    links = message.get("link") if isinstance(message, dict) else None
    if not isinstance(links, list):
        return []
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    for link in links:
        if not isinstance(link, dict):
            continue
        if link.get("content-type") != "application/pdf":
            continue
        if link.get("intended-application") not in _CROSSREF_TDM_INTENDED:
            continue
        url = link.get("URL")
        if isinstance(url, str) and url and url not in seen:
            candidates.append(OaCandidate(url=url, tier=OaTier.PUBLISHER))
            seen.add(url)
    return candidates


#: ``openAccessPdf.status`` values that mean the copy is publisher-hosted. Anything
#: else (``GREEN``, unknown, absent) is ranked as a repository copy: over-ranking an
#: unknown host would put an unverified URL ahead of a genuine version of record.
_S2_PUBLISHER_STATUSES = frozenset({"gold", "hybrid", "bronze"})


def semantic_scholar_oa_candidates(payload: Any) -> list[OaCandidate]:
    """The OA PDF candidate (if any) from one Semantic Scholar paper record.

    Shape verified live 2026-07-29: ``openAccessPdf`` is an object with ``url``,
    ``status`` (e.g. ``"GREEN"``) and ``license``.

    Args:
        payload: The raw ``graph/v1/paper/DOI:{doi}`` response payload.

    Returns:
        At most one candidate; empty for unusable shapes.
    """
    if not isinstance(payload, dict):
        return []
    open_access_pdf = payload.get("openAccessPdf")
    if not isinstance(open_access_pdf, dict):
        return []
    url = open_access_pdf.get("url")
    if not isinstance(url, str) or not url:
        return []
    status = open_access_pdf.get("status")
    tier = (
        OaTier.PUBLISHER if isinstance(status, str) and status.lower() in _S2_PUBLISHER_STATUSES else OaTier.REPOSITORY
    )
    return [OaCandidate(url=url, tier=tier)]


def core_oa_candidates(payload: Any, *, doi: str) -> list[OaCandidate]:
    """Aggregated repository copies from one CORE v3 works search response.

    UNVERIFIED SHAPE: CORE requires an API key, so this parser could not be checked
    against a live response; it follows the published v3 work schema (``results[]``
    with ``downloadUrl``, ``links[]`` of type ``download``, ``sourceFulltextUrls``).
    Defensive on every field, like all parsers in this module.

    A result is accepted ONLY when it names the wanted DOI itself: CORE's search is
    fuzzy, and a topically-similar work's PDF is the wrong paper -- far worse than a
    miss.

    Args:
        payload: The raw CORE search response payload, of unknown shape.
        doi: The wanted paper's bare normalized DOI.

    Returns:
        Deduplicated :attr:`OaTier.REPOSITORY` candidates; empty for unusable shapes.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    for work in results:
        if not isinstance(work, dict):
            continue
        if normalize_doi(work.get("doi")) != doi:
            continue
        urls: list[Any] = [work.get("downloadUrl")]
        links = work.get("links")
        if isinstance(links, list):
            urls.extend(link.get("url") for link in links if isinstance(link, dict) and link.get("type") == "download")
        source_urls = work.get("sourceFulltextUrls")
        if isinstance(source_urls, list):
            urls.extend(source_urls)
        for url in urls:
            if isinstance(url, str) and url and url not in seen:
                candidates.append(OaCandidate(url=url, tier=OaTier.REPOSITORY))
                seen.add(url)
    return candidates


def doaj_oa_candidates(payload: Any, *, doi: str) -> list[OaCandidate]:
    """Gold-OA journal full-text links from one DOAJ article search response.

    Everything DOAJ indexes is a fully open-access journal, so its ``fulltext`` links
    point at the publisher's version of record (:attr:`OaTier.PUBLISHER`); links whose
    ``content_type`` is ``PDF`` are ordered before the rest because the fetch side
    treats an HTML landing page as NOT_A_DOCUMENT. A result is accepted only when its
    ``bibjson.identifier`` actually names the wanted DOI.

    Shape per the published DOAJ API data model (``results[].bibjson.link[]`` with
    ``type``/``url``/``content_type``); a live probe was blocked (HTTP 403 to
    non-browser fetchers), so treat as UNVERIFIED against a live response.

    Args:
        payload: The raw DOAJ search response payload, of unknown shape.
        doi: The wanted paper's bare normalized DOI.

    Returns:
        Deduplicated :attr:`OaTier.PUBLISHER` candidates, PDFs first.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    pdf_links: list[OaCandidate] = []
    other_links: list[OaCandidate] = []
    seen: set[str] = set()
    for result in results:
        bibjson = result.get("bibjson") if isinstance(result, dict) else None
        if not isinstance(bibjson, dict):
            continue
        if not _doaj_names_doi(bibjson.get("identifier"), doi):
            continue
        links = bibjson.get("link")
        if not isinstance(links, list):
            continue
        for link in links:
            if not isinstance(link, dict) or link.get("type") != "fulltext":
                continue
            url = link.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            content_type = link.get("content_type")
            is_pdf = isinstance(content_type, str) and content_type.upper() == "PDF"
            (pdf_links if is_pdf else other_links).append(OaCandidate(url=url, tier=OaTier.PUBLISHER))
    return pdf_links + other_links


def _doaj_names_doi(identifiers: Any, doi: str) -> bool:
    """True when a DOAJ ``bibjson.identifier`` list names exactly the wanted DOI.

    Args:
        identifiers: The raw identifier list, of unknown shape.
        doi: The wanted paper's bare normalized DOI.

    Returns:
        Whether the DOI is named. A result carrying no DOI identifier at all is
        REJECTED: the query was DOI-keyed, but only the record's own claim proves the
        hit is not a fuzzy-match stranger.
    """
    if not isinstance(identifiers, list):
        return False
    for identifier in identifiers:
        if not isinstance(identifier, dict) or identifier.get("type") != "doi":
            continue
        if normalize_doi(identifier.get("id")) == doi:
            return True
    return False


def chemrxiv_oa_candidates(payload: Any, *, doi: str, title: str | None) -> list[OaCandidate]:
    """Preprint PDFs from a ChemRxiv (Cambridge Open Engage) term-search response.

    An item is accepted ONLY on a strong match: its own preprint DOI equals the wanted
    DOI, its ``vor.vorDoi`` (the published version's DOI) equals the wanted DOI, or its
    title matches the wanted title exactly (case/punctuation-insensitive). Term search
    returns topically-similar preprints, and fetching a similar-but-wrong preprint is
    far worse than a miss.

    Shape per the community clients of the Open Engage API (``itemHits[].item`` with
    ``doi``, ``title``, ``vor.vorDoi``, ``asset.original.url``); a live probe was
    blocked (HTTP 403 to non-browser fetchers), so treat ``asset.original.url`` in
    particular as UNVERIFIED against a live response.

    The ``public-api/v1/items?term=`` endpoint returned 403/404 for all 12 probed
    titles on 2026-07-29; needs re-verification before re-enabling by default (see
    :data:`DEFAULT_ENABLED_PROVIDERS`).

    Args:
        payload: The raw items search payload, of unknown shape.
        doi: The wanted paper's bare normalized DOI.
        title: The wanted paper's title, if known.

    Returns:
        Deduplicated :attr:`OaTier.PREPRINT` candidates; empty for unusable shapes.
    """
    hits = payload.get("itemHits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        return []
    wanted_title = _normalized_title(title) if title else ""
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    for hit in hits:
        item = hit.get("item") if isinstance(hit, dict) else None
        if not isinstance(item, dict) or not _chemrxiv_item_matches(item, doi=doi, wanted_title=wanted_title):
            continue
        asset = item.get("asset")
        original = asset.get("original") if isinstance(asset, dict) else None
        url = original.get("url") if isinstance(original, dict) else None
        if isinstance(url, str) and url and url not in seen:
            candidates.append(OaCandidate(url=url, tier=OaTier.PREPRINT))
            seen.add(url)
    return candidates


def _chemrxiv_item_matches(item: dict[str, Any], *, doi: str, wanted_title: str) -> bool:
    """Strong-match test for one ChemRxiv item against the wanted paper."""
    if normalize_doi(item.get("doi")) == doi:
        return True
    vor = item.get("vor")
    if isinstance(vor, dict) and normalize_doi(vor.get("vorDoi")) == doi:
        return True
    if wanted_title:
        item_title = item.get("title")
        if isinstance(item_title, str) and _normalized_title(item_title) == wanted_title:
            return True
    return False


_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def arxiv_oa_candidates(atom_xml: str, *, doi: str, title: str | None) -> list[OaCandidate]:
    """Preprint PDFs from an arXiv Atom query response.

    Shape verified live 2026-07-29: each ``entry`` carries Atom ``id``/``title``, an
    OPTIONAL ``arxiv:doi`` (namespace ``http://arxiv.org/schemas/atom``; present only
    when the authors linked the published version), and a PDF ``link`` with
    ``title="pdf"`` / ``type="application/pdf"``. An entry is accepted ONLY when its
    ``arxiv:doi`` equals the wanted DOI or its title matches exactly
    (case/punctuation-insensitive): arXiv title search returns similar papers, and a
    wrong-paper match is far worse than a miss.

    Args:
        atom_xml: The raw Atom response body, decoded as text.
        doi: The wanted paper's bare normalized DOI.
        title: The wanted paper's title, if known.

    Returns:
        Deduplicated :attr:`OaTier.PREPRINT` candidates; empty for malformed feeds.
    """
    if not atom_xml or not atom_xml.strip():
        return []
    try:
        root = ElementTree.fromstring(atom_xml)  # noqa: S314 - budget-capped body from a fixed, trusted endpoint
    except ElementTree.ParseError:
        logger.warning("arXiv response was not parseable Atom XML; no candidates")
        return []
    wanted_title = _normalized_title(title) if title else ""
    candidates: list[OaCandidate] = []
    seen: set[str] = set()
    for entry in root.iter(f"{_ATOM_NS}entry"):
        entry_doi = normalize_doi(entry.findtext(f"{_ARXIV_NS}doi"))
        entry_title = entry.findtext(f"{_ATOM_NS}title") or ""
        doi_match = entry_doi is not None and entry_doi == doi
        title_match = bool(wanted_title) and _normalized_title(entry_title) == wanted_title
        if not (doi_match or title_match):
            continue
        for link in entry.iter(f"{_ATOM_NS}link"):
            if link.get("type") != "application/pdf" and link.get("title") != "pdf":
                continue
            href = link.get("href")
            if isinstance(href, str) and href and href not in seen:
                candidates.append(OaCandidate(url=href, tier=OaTier.PREPRINT))
                seen.add(href)
    return candidates


@dataclass(frozen=True)
class OaResolution:
    """The outcome of one DOI's open-access resolution.

    ``note`` is a human-readable account of what resolution actually did (which
    indexes were consulted, which were skipped and why, how many candidates each
    advertised, and whether a candidate is a preprint rather than the version of
    record). It exists so a paper queued with
    :attr:`~carmel.schemas.acquisition.AcquisitionReason.NO_OPEN_ACCESS_COPY` can
    show the operator an honest ``detail`` -- "skipped: consent withheld" and "no
    index advertises anything" are very different facts and must not collapse into
    one wording.
    """

    candidates: tuple[str, ...]
    """OA PDF URLs to try fetching, best first, deduplicated across indexes."""
    note: str
    complete: bool = True
    """Whether every enabled provider actually got to answer.

    ``False`` means resolution was CUT SHORT -- the per-paper lookup cap was reached, or
    a provider's lookup failed in transit (timeout, HTTP error) -- so an empty
    ``candidates`` establishes nothing about whether an OA copy exists. A caller must
    then queue the paper as
    :attr:`~carmel.schemas.acquisition.AcquisitionReason.OA_LOOKUP_INCOMPLETE`, never as
    ``NO_OPEN_ACCESS_COPY``: "we did not finish looking" is not "there is nothing there".
    This is the same asserted-vs-observed distinction already fixed once for
    ``PAYWALLED``; it must not creep back in one level down.

    A provider declining via :class:`_ProviderSkip` (a missing optional API key, say)
    deliberately does NOT clear this flag. That is a stable configuration fact, already
    spelled out in ``note``, not a transient unknown -- and since CORE ships keyless by
    default, counting it as incomplete would mark every paper incomplete and make the
    distinction worthless.
    """


class OpenAccessResolverProtocol(Protocol):
    """Structural type for anything that can resolve a DOI to OA PDF candidates."""

    def resolve(self, doi: str, *, title: str | None = None) -> OaResolution: ...


class _ProviderSkip(Exception):
    """Raised by a provider lookup to decline WITHOUT a network call.

    The message becomes the human-readable skip reason in the resolution note
    ("CORE: skipped (no API key configured)"). Deliberately not a failure: a skip is
    a configuration fact, a :class:`SearchError` is a transport outcome.
    """


class OpenAccessResolver(_KeylessSearchTool):
    """DOI -> open-access PDF candidates, via a pluggable list of OA index providers.

    Providers registered (each at most one lookup call): OpenAlex, Unpaywall, Crossref
    (publisher TDM links), Semantic Scholar, CORE, DOAJ, ChemRxiv and arXiv, consulted
    in that order for whichever of them are enabled -- see
    :data:`DEFAULT_ENABLED_PROVIDERS` (ChemRxiv is registered but disabled by
    default). Adding a provider is a small, isolated change: one pure parser, one
    ``_lookup_*`` method, one entry in the registry construction, one name in
    :data:`DEFAULT_ENABLED_PROVIDERS` if it should be queried by default.

    Every provider is independently fail-soft -- one index erroring, rate-limiting or
    returning junk never loses the others' candidates and never aborts the run -- and
    every lookup goes through :func:`~carmel.agents.tools.search.budgeted_get_raw` /
    ``budgeted_get_json``, so budget reservation happens before any socket opens and
    the ``external_provider_consent`` gate applies. Belt and braces: :meth:`resolve`
    also refuses up front when consent is withheld, making a consent-less run provably
    network-silent. Total lookup calls per paper are hard-capped at
    :data:`MAX_OA_LOOKUP_CALLS_PER_PAPER`.

    Credential-gated providers skip cleanly (one warning per resolver, not one per
    paper) instead of failing: Unpaywall REQUIRES a contact email as a condition of
    use (a fake or placeholder address is never sent), and CORE REQUIRES an API key
    (sent only in the ``Authorization`` header, never in a URL, a log line or a
    resolution note).
    """

    def __init__(
        self,
        *,
        ledger: BudgetLedger,
        external_provider_consent: bool,
        contact_email: str | None = None,
        unpaywall_email: str | None = None,
        core_api_key: str | None = None,
        semantic_scholar_api_key: str | None = None,
        opener: Callable[..., Any] | None = None,
        timeout_s: float = OA_INDEX_LOOKUP_TIMEOUT_S,
    ) -> None:
        """Construct the resolver.

        Args:
            ledger: Budget ledger; every lookup is reserved and settled through it.
            external_provider_consent: Operator opt-in to third-party network egress.
            contact_email: Optional OpenAlex/Crossref "polite pool" address (``mailto=``).
            unpaywall_email: Contact email REQUIRED by Unpaywall; ``None`` skips
                Unpaywall entirely rather than sending a fabricated address.
            core_api_key: API key REQUIRED by CORE; ``None`` skips CORE entirely.
                Travels only in the ``Authorization`` header and is never logged.
            semantic_scholar_api_key: OPTIONAL Semantic Scholar key (higher rate
                limits); ``None`` still queries their keyless shared pool.
            opener: Injected ``(url, headers=..., timeout_s=...) -> response`` opener,
                for tests. Defaults to a real urllib GET.
            timeout_s: Per-request socket timeout for the index/metadata lookup calls
                this resolver makes. Defaults to :data:`OA_INDEX_LOOKUP_TIMEOUT_S`
                (10s), much shorter than the ~30s used elsewhere for actual PDF/
                document fetches, because these are small JSON/Atom calls and a dead
                index should not cost more than a few seconds per paper.
        """
        super().__init__(
            ledger=ledger,
            external_provider_consent=external_provider_consent,
            contact_email=contact_email,
            opener=opener,
            timeout_s=timeout_s,
        )
        self._unpaywall_email = unpaywall_email
        self._core_api_key = core_api_key
        self._semantic_scholar_api_key = semantic_scholar_api_key
        self._warned_skips: set[str] = set()
        self._warned_failures: set[str] = set()
        self._lookup_calls = 0
        #: The provider registry. Order is the consultation order and, within a tier,
        #: the candidate order. To add a provider: write a pure parser above, a
        #: ``_lookup_*`` method below, and register it here.
        #:
        #: FatCat / Internet Archive Scholar was removed entirely on 2026-07-29
        #: because api.fatcat.wiki no longer resolves (connection timeouts on every
        #: lookup) -- do not re-add from the old documented schema without
        #: re-verifying the endpoint is alive.
        #:
        #: Only providers named in :data:`DEFAULT_ENABLED_PROVIDERS` are actually
        #: queried by default; see that constant for why ChemRxiv is registered but
        #: excluded.
        all_providers: tuple[tuple[str, Callable[[str, str | None], list[OaCandidate]]], ...] = (
            ("OpenAlex", self._lookup_openalex),
            ("Unpaywall", self._lookup_unpaywall),
            ("Crossref", self._lookup_crossref),
            ("Semantic Scholar", self._lookup_semantic_scholar),
            ("CORE", self._lookup_core),
            ("DOAJ", self._lookup_doaj),
            ("ChemRxiv", self._lookup_chemrxiv),
            ("arXiv", self._lookup_arxiv),
        )
        self._providers = tuple(p for p in all_providers if p[0] in DEFAULT_ENABLED_PROVIDERS)

    def resolve(self, doi: str, *, title: str | None = None) -> OaResolution:
        """Resolve one DOI (and optionally a title) to fetchable OA PDF candidates.

        Args:
            doi: Bare, normalized DOI (``10.xxxx/yyy``).
            title: The paper's title, if known. Only the title-matched providers
                (ChemRxiv, arXiv) use it; without a title they are skipped, because
                a weak match there would fetch the WRONG paper.

        Returns:
            Candidates ordered publisher OA PDF first, then repository/archive
            copies, then preprints strictly last (a preprint is not the version of
            record, and the note says so), deduplicated across providers -- plus an
            honest note of what each provider did. One provider failing never loses
            the others' candidates; a failure costs exactly one attempt (no retry).

        Raises:
            BudgetExceededError: Propagated from the ledger; a budget ceiling is a
                run-level condition, not a per-paper resolution outcome.
        """
        if not self._external_provider_consent:
            return OaResolution(
                candidates=(),
                note="open-access resolution skipped: external_provider_consent is False",
            )

        notes: list[str] = []
        ranked: list[OaCandidate] = []
        self._lookup_calls = 0
        # Cleared the moment a provider is cut short or fails in transit, so an empty
        # result is never reported as "no OA copy exists" (see OaResolution.complete).
        complete = True

        for name, lookup in self._providers:
            if self._lookup_calls >= MAX_OA_LOOKUP_CALLS_PER_PAPER:
                notes.append(f"{name}: skipped (per-paper lookup cap of {MAX_OA_LOOKUP_CALLS_PER_PAPER} calls reached)")
                complete = False
                continue
            try:
                found = lookup(doi, title)
            except _ProviderSkip as skip:
                notes.append(f"{name}: skipped ({skip})")
                continue
            except SearchError as exc:
                # Warn at most once per provider per resolver instance -- not once per
                # paper -- so a dead index (e.g. a timing-out endpoint) costs one log
                # line for the whole run, not one per paper resolved.
                if name not in self._warned_failures:
                    self._warned_failures.add(name)
                    logger.warning("%s OA lookup failed: %s", name, exc)
                notes.append(f"{name}: lookup failed ({exc})")
                complete = False
                continue
            plural = "" if len(found) == 1 else "s"
            note = f"{name}: {len(found)} OA PDF candidate{plural}"
            if found and all(candidate.tier is OaTier.PREPRINT for candidate in found):
                note += " (preprint, not the version of record)"
            notes.append(note)
            ranked.extend(found)

        # Stable sort: tiers order globally (publisher, repository, preprint) while
        # provider order and each provider's own ordering are preserved within a
        # tier. Dedupe keeps the first -- and therefore best-tiered -- occurrence.
        candidates: list[str] = []
        for candidate in sorted(ranked, key=lambda c: c.tier):
            if candidate.url not in candidates:
                candidates.append(candidate.url)

        return OaResolution(candidates=tuple(candidates), note="; ".join(notes), complete=complete)

    # ------------------------- provider lookups -------------------------

    def _lookup_openalex(self, doi: str, title: str | None) -> list[OaCandidate]:
        url = f"{OPENALEX_ENDPOINT}/doi:{quote(doi, safe='/')}"
        if self._contact_email:
            url = f"{url}?mailto={quote_plus(self._contact_email)}"
        return openalex_oa_candidates(self._counted_get_json(url))

    def _lookup_unpaywall(self, doi: str, title: str | None) -> list[OaCandidate]:
        if not self._unpaywall_email:
            self._warn_skip_once(
                "unpaywall",
                "Unpaywall lookups are skipped: no contact email configured "
                "(set agents.unpaywall_email or the CARMEL_UNPAYWALL_EMAIL "
                "environment variable); Unpaywall requires a real address and "
                "Carmel will not send a fabricated one",
            )
            raise _ProviderSkip("no contact email configured")
        url = f"{UNPAYWALL_ENDPOINT}/{quote(doi, safe='/')}?email={quote_plus(self._unpaywall_email)}"
        return unpaywall_oa_candidates(self._counted_get_json(url))

    def _lookup_crossref(self, doi: str, title: str | None) -> list[OaCandidate]:
        url = f"{CROSSREF_ENDPOINT}/{quote(doi, safe='/')}"
        if self._contact_email:
            url = f"{url}?mailto={quote_plus(self._contact_email)}"
        return crossref_tdm_candidates(self._counted_get_json(url))

    def _lookup_semantic_scholar(self, doi: str, title: str | None) -> list[OaCandidate]:
        url = f"{SEMANTIC_SCHOLAR_ENDPOINT}/DOI:{quote(doi, safe='/')}?fields=openAccessPdf,externalIds,title"
        headers = {"x-api-key": self._semantic_scholar_api_key} if self._semantic_scholar_api_key else None
        return semantic_scholar_oa_candidates(self._counted_get_json(url, extra_headers=headers))

    def _lookup_core(self, doi: str, title: str | None) -> list[OaCandidate]:
        if not self._core_api_key:
            self._warn_skip_once(
                "core",
                "CORE lookups are skipped: no API key configured (set "
                "agents.core_api_key or the CARMEL_CORE_API_KEY environment "
                "variable; a free key is available from core.ac.uk)",
            )
            raise _ProviderSkip("no API key configured")
        query = quote_plus(f'doi:"{doi}"')
        url = f"{CORE_ENDPOINT}?q={query}&limit={_PROVIDER_SEARCH_LIMIT}"
        payload = self._counted_get_json(url, extra_headers={"Authorization": f"Bearer {self._core_api_key}"})
        return core_oa_candidates(payload, doi=doi)

    def _lookup_doaj(self, doi: str, title: str | None) -> list[OaCandidate]:
        url = f"{DOAJ_ENDPOINT}/{quote(f'doi:{doi}', safe='')}"
        return doaj_oa_candidates(self._counted_get_json(url), doi=doi)

    def _lookup_chemrxiv(self, doi: str, title: str | None) -> list[OaCandidate]:
        if not title:
            raise _ProviderSkip("no title to match against")
        url = f"{CHEMRXIV_ENDPOINT}?term={quote_plus(title)}&limit={_PROVIDER_SEARCH_LIMIT}"
        return chemrxiv_oa_candidates(self._counted_get_json(url), doi=doi, title=title)

    def _lookup_arxiv(self, doi: str, title: str | None) -> list[OaCandidate]:
        if not title:
            raise _ProviderSkip("no title to match against")
        query = quote_plus(f'ti:"{title}"')
        url = f"{ARXIV_ENDPOINT}?search_query={query}&max_results={_PROVIDER_SEARCH_LIMIT}"
        return arxiv_oa_candidates(self._counted_get_text(url), doi=doi, title=title)

    # ------------------------- shared plumbing -------------------------

    def _warn_skip_once(self, key: str, message: str) -> None:
        """Log a provider-skip warning once per resolver, not once per paper."""
        if key not in self._warned_skips:
            self._warned_skips.add(key)
            logger.warning(message)

    def _counted_get_json(self, url: str, extra_headers: dict[str, str] | None = None) -> Any | None:
        """One budgeted, cap-counted JSON lookup."""
        self._lookup_calls += 1
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return budgeted_get_json(
            url,
            headers=headers,
            ledger=self._ledger,
            opener=self._opener,
            timeout_s=self._timeout_s,
            external_provider_consent=self._external_provider_consent,
        )

    def _counted_get_text(self, url: str, accept: str = "application/atom+xml") -> str:
        """One budgeted, cap-counted lookup returning the body as text (arXiv Atom)."""
        self._lookup_calls += 1
        raw = budgeted_get_raw(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": accept},
            ledger=self._ledger,
            opener=self._opener,
            timeout_s=self._timeout_s,
            external_provider_consent=self._external_provider_consent,
        )
        return raw.decode("utf-8", errors="replace")
