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
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, quote_plus

from carmel.agents.budget import BudgetLedger
from carmel.agents.tools.search import SearchError, SearchResult, budgeted_get_json, default_opener
from carmel.logger import get_logger

logger = get_logger("agents.tools.academic")

OPENALEX_ENDPOINT = "https://api.openalex.org/works"
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2"

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


def openalex_oa_pdf_urls(work: Any) -> list[str]:
    """Ordered OA PDF candidates from one OpenAlex work record.

    ``best_oa_location.pdf_url`` comes first (for gold OA that is the publisher's
    version of record), followed by every ``locations[].pdf_url`` whose location is
    explicitly ``is_oa: true``. A location's pdf_url WITHOUT the OA flag is excluded:
    using it would assert open access the index never claimed.

    Args:
        work: The raw OpenAlex work payload, of unknown/untrusted shape.

    Returns:
        Deduplicated candidate URLs in preference order; empty for unusable shapes.
    """
    if not isinstance(work, dict):
        return []
    urls: list[str] = []
    best = work.get("best_oa_location")
    if isinstance(best, dict):
        pdf_url = best.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url:
            urls.append(pdf_url)
    locations = work.get("locations")
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict) or location.get("is_oa") is not True:
                continue
            pdf_url = location.get("pdf_url")
            if isinstance(pdf_url, str) and pdf_url and pdf_url not in urls:
                urls.append(pdf_url)
    return urls


def unpaywall_oa_pdf_urls(payload: Any) -> list[str]:
    """Ordered OA PDF candidates from one Unpaywall DOI record.

    ``best_oa_location.url_for_pdf`` first, then every ``oa_locations[].url_for_pdf``.
    Everything in ``oa_locations`` is open access by construction of the API.

    Args:
        payload: The raw Unpaywall response payload, of unknown/untrusted shape.

    Returns:
        Deduplicated candidate URLs in preference order; empty for unusable shapes.
    """
    if not isinstance(payload, dict):
        return []
    urls: list[str] = []
    best = payload.get("best_oa_location")
    if isinstance(best, dict):
        url_for_pdf = best.get("url_for_pdf")
        if isinstance(url_for_pdf, str) and url_for_pdf:
            urls.append(url_for_pdf)
    oa_locations = payload.get("oa_locations")
    if isinstance(oa_locations, list):
        for location in oa_locations:
            if not isinstance(location, dict):
                continue
            url_for_pdf = location.get("url_for_pdf")
            if isinstance(url_for_pdf, str) and url_for_pdf and url_for_pdf not in urls:
                urls.append(url_for_pdf)
    return urls


@dataclass(frozen=True)
class OaResolution:
    """The outcome of one DOI's open-access resolution.

    ``note`` is a human-readable account of what resolution actually did (which
    indexes were consulted, which were skipped and why, how many candidates each
    advertised). It exists so a paper queued with
    :attr:`~carmel.schemas.acquisition.AcquisitionReason.NO_OPEN_ACCESS_COPY` can
    show the operator an honest ``detail`` -- "skipped: consent withheld" and "both
    indexes advertise nothing" are very different facts and must not collapse into
    one wording.
    """

    candidates: tuple[str, ...]
    """OA PDF URLs to try fetching, best first, deduplicated across indexes."""
    note: str


class OpenAccessResolverProtocol(Protocol):
    """Structural type for anything that can resolve a DOI to OA PDF candidates."""

    def resolve(self, doi: str) -> OaResolution: ...


class OpenAccessResolver(_KeylessSearchTool):
    """DOI -> open-access PDF candidates, via OpenAlex and Unpaywall.

    Both lookups go through :func:`~carmel.agents.tools.search.budgeted_get_json`
    (inherited ``_get``), so budget reservation happens before any socket opens and
    the ``external_provider_consent`` gate applies -- belt and braces, since
    :meth:`resolve` also refuses up front when consent is withheld, making a
    consent-less run provably network-silent.

    Unpaywall REQUIRES a contact email as a condition of use. When none is
    configured the Unpaywall lookup is skipped entirely (one warning per resolver,
    not one per paper); a fake or placeholder address is never sent.
    """

    def __init__(
        self,
        *,
        ledger: BudgetLedger,
        external_provider_consent: bool,
        contact_email: str | None = None,
        unpaywall_email: str | None = None,
        opener: Callable[..., Any] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        """Construct the resolver.

        Args:
            ledger: Budget ledger; every lookup is reserved and settled through it.
            external_provider_consent: Operator opt-in to third-party network egress.
            contact_email: Optional OpenAlex "polite pool" address (``mailto=``).
            unpaywall_email: Contact email REQUIRED by Unpaywall; ``None`` skips
                Unpaywall entirely rather than sending a fabricated address.
            opener: Injected ``(url, headers=..., timeout_s=...) -> response`` opener,
                for tests. Defaults to a real urllib GET.
            timeout_s: Per-request socket timeout.
        """
        super().__init__(
            ledger=ledger,
            external_provider_consent=external_provider_consent,
            contact_email=contact_email,
            opener=opener,
            timeout_s=timeout_s,
        )
        self._unpaywall_email = unpaywall_email
        self._warned_unpaywall_skipped = False

    def resolve(self, doi: str) -> OaResolution:
        """Resolve one DOI to fetchable OA PDF candidates.

        Args:
            doi: Bare, normalized DOI (``10.xxxx/yyy``).

        Returns:
            Candidates in preference order -- OpenAlex first (its
            ``best_oa_location`` leads, so a publisher OA PDF precedes repository
            copies), then any Unpaywall-only URLs -- plus an honest note of what was
            consulted. One index failing never loses the other's candidates.

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
        candidates: list[str] = []

        openalex_url = f"{OPENALEX_ENDPOINT}/doi:{quote(doi, safe='/')}"
        if self._contact_email:
            openalex_url = f"{openalex_url}?mailto={quote_plus(self._contact_email)}"
        for url in self._lookup(openalex_url, "OpenAlex", openalex_oa_pdf_urls, notes):
            if url not in candidates:
                candidates.append(url)

        if not self._unpaywall_email:
            if not self._warned_unpaywall_skipped:
                self._warned_unpaywall_skipped = True
                logger.warning(
                    "Unpaywall lookups are skipped: no contact email configured "
                    "(set agents.unpaywall_email or the CARMEL_UNPAYWALL_EMAIL "
                    "environment variable); Unpaywall requires a real address and "
                    "Carmel will not send a fabricated one"
                )
            notes.append("Unpaywall: skipped (no contact email configured)")
        else:
            unpaywall_url = f"{UNPAYWALL_ENDPOINT}/{quote(doi, safe='/')}?email={quote_plus(self._unpaywall_email)}"
            for url in self._lookup(unpaywall_url, "Unpaywall", unpaywall_oa_pdf_urls, notes):
                if url not in candidates:
                    candidates.append(url)

        return OaResolution(candidates=tuple(candidates), note="; ".join(notes))

    def _lookup(
        self,
        url: str,
        index_name: str,
        parse: Callable[[Any], list[str]],
        notes: list[str],
    ) -> list[str]:
        """One budgeted index lookup; transport failure yields no candidates, loudly.

        Args:
            url: Fully-formed lookup URL.
            index_name: Human-readable index name for the note.
            parse: Payload -> candidate URLs parser.
            notes: Accumulator the outcome sentence is appended to.

        Returns:
            The parsed candidates (empty on failure/unusable payload).
        """
        try:
            payload = self._get(url)
        except SearchError as exc:
            logger.warning("%s OA lookup failed: %s", index_name, exc)
            notes.append(f"{index_name}: lookup failed ({exc})")
            return []
        urls = parse(payload)
        plural = "" if len(urls) == 1 else "s"
        notes.append(f"{index_name}: {len(urls)} OA PDF candidate{plural}")
        return urls
