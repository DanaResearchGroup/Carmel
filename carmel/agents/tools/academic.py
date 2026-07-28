"""Keyless scholarly search backends: OpenAlex and Crossref.

These exist because the generic :class:`~carmel.agents.tools.search.HttpSearchTool`
assumes an operator-supplied endpoint plus a bearer key, which no scholarly index
requires — and demanding one made them impossible to configure at all.

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
from typing import Any
from urllib.parse import quote_plus

from carmel.agents.budget import BudgetLedger
from carmel.agents.tools.search import SearchResult, budgeted_get_json, default_opener
from carmel.logger import get_logger

logger = get_logger("agents.tools.academic")

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"

#: Sent on every request so the upstream operator can identify (and contact us about)
#: this client rather than silently rate-limiting it.
USER_AGENT = "Carmel/0.1 (+https://github.com/DanaResearchGroup/Carmel)"

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
        full-text URL. Repository-hosted copies win over publisher-hosted ones.
    """
    if not isinstance(locations, list):
        return None, ""

    best_rank = len(_HOST_TYPE_PREFERENCE)
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
        contact_email: str | None = None,
        opener: Callable[..., Any] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Construct the tool.

        Args:
            ledger: Budget ledger; every search call is reserved and settled through it.
            contact_email: Optional address for the API's "polite pool" (higher rate
                limits). Omitted from the URL entirely when not supplied.
            opener: Injected ``(url, headers=..., timeout_s=...) -> response`` opener,
                for tests. Defaults to a real urllib GET.
            timeout_s: Per-request socket timeout.
        """
        self._ledger = ledger
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
