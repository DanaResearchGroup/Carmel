# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the keyless OpenAlex/Crossref search adapters.

Payload fixtures are trimmed copies of REAL responses observed during a live probe of
combustion-kinetics queries, including the awkward cases that probe surfaced: an
``is_oa`` work with no fetchable location, a publisher-hosted copy competing with a
repository-hosted one, and Crossref returning supplementary-material components
alongside the article they belong to.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

import carmel.agents.tools.academic as academic
from carmel.agents.budget import BudgetLedger
from carmel.agents.tools.academic import (
    ARXIV_ENDPOINT,
    CHEMRXIV_ENDPOINT,
    CORE_ENDPOINT,
    CROSSREF_ENDPOINT,
    DOAJ_ENDPOINT,
    ELSEVIER_ARTICLE_ENDPOINT,
    EUROPEPMC_ENDPOINT,
    OPENALEX_ENDPOINT,
    SEMANTIC_SCHOLAR_ENDPOINT,
    UNPAYWALL_ENDPOINT,
    CrossrefSearchTool,
    OaCandidate,
    OaLookupCoverage,
    OaTier,
    OpenAccessResolver,
    OpenAlexSearchTool,
    arxiv_oa_candidates,
    chemrxiv_oa_candidates,
    core_oa_candidates,
    crossref_tdm_candidates,
    dedupe_by_doi,
    doaj_oa_candidates,
    elsevier_oa_candidates,
    europepmc_oa_candidates,
    normalize_doi,
    openalex_oa_pdf_urls,
    semantic_scholar_oa_candidates,
    unpaywall_oa_pdf_urls,
)
from carmel.agents.tools.search import SearchResult
from carmel.config import AgentBudgetConfig


class _FakeResponse:
    """Minimal file-like HTTP response over a fixed payload.

    ``str``/``bytes`` payloads are served verbatim (the arXiv adapter reads Atom XML,
    and a raw string is also how a malformed-JSON body is simulated); anything else is
    JSON-encoded.
    """

    def __init__(self, payload: Any) -> None:
        if isinstance(payload, bytes):
            self._data = payload
        elif isinstance(payload, str):
            self._data = payload.encode("utf-8")
        else:
            self._data = json.dumps(payload).encode("utf-8")
        self._offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk, self._offset = self._data[self._offset :], len(self._data)
            return chunk
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


def _opener_for(payload: Any, seen: list[str] | None = None) -> Any:
    """Build an injected opener returning ``payload``, recording requested URLs."""

    def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> _FakeResponse:
        if seen is not None:
            seen.append(url)
        return _FakeResponse(payload)

    return opener


@pytest.fixture
def ledger() -> BudgetLedger:
    return BudgetLedger(AgentBudgetConfig())


class TestNormalizeDoi:
    """DOIs arrive in three spellings across the two backends; all must collapse."""

    @pytest.mark.parametrize(
        "raw",
        [
            "https://doi.org/10.1016/J.CombustFlame.2015.11.011",
            "http://dx.doi.org/10.1016/j.combustflame.2015.11.011",
            "doi:10.1016/j.combustflame.2015.11.011",
            "10.1016/j.combustflame.2015.11.011",
            "  10.1016/J.combustflame.2015.11.011  ",
        ],
    )
    def test_every_spelling_collapses_to_the_bare_lowercase_form(self, raw: str) -> None:
        assert normalize_doi(raw) == "10.1016/j.combustflame.2015.11.011"

    @pytest.mark.parametrize("raw", [None, 42, "", "not-a-doi", "https://example.org/paper"])
    def test_non_dois_are_rejected_rather_than_mangled(self, raw: Any) -> None:
        assert normalize_doi(raw) is None


class TestOpenAlexSearchTool:
    def test_prefers_the_repository_copy_over_the_publisher_copy(self, ledger: BudgetLedger) -> None:
        """The live probe found publisher-hosted PDFs return 403 to non-browser clients
        while repository copies serve real bytes; preferring the repository is what
        lifted the end-to-end success rate off zero, so it must not regress."""
        payload = {
            "results": [
                {
                    "title": "Shock tube ignition delay of methane",
                    "doi": "https://doi.org/10.1016/j.combustflame.2015.11.011",
                    "open_access": {"is_oa": True},
                    "locations": [
                        {
                            "pdf_url": "https://publisher.example.com/paper.pdf",
                            "host_type": "publisher",
                            "source": {"display_name": "Elsevier"},
                        },
                        {
                            "pdf_url": "https://osti.example.gov/paper.pdf",
                            "host_type": "repository",
                            "source": {"display_name": "OSTI"},
                        },
                    ],
                }
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("methane ignition delay")

        assert len(results) == 1
        assert results[0].pdf_url == "https://osti.example.gov/paper.pdf"
        assert results[0].repository == "OSTI"

    def test_open_access_work_with_no_fetchable_location_still_returned_without_pdf(self, ledger: BudgetLedger) -> None:
        """48% of probed works were flagged is_oa but only 18% carried a full-text URL.
        Such a work is still a real, citable paper -- it must be returned (so it can be
        queued for manual acquisition), but must NOT claim a pdf_url it does not have."""
        payload = {
            "results": [
                {
                    "title": "Laminar flame speed of ammonia blends",
                    "doi": "https://doi.org/10.1016/j.proci.2020.06.197",
                    "open_access": {"is_oa": True},
                    "locations": [{"host_type": "publisher", "source": {"display_name": "Elsevier"}}],
                }
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("ammonia flame speed")

        assert len(results) == 1
        assert results[0].pdf_url is None
        assert results[0].is_open_access is True
        assert results[0].url == "https://doi.org/10.1016/j.proci.2020.06.197"

    def test_a_record_about_a_paper_is_not_returned_as_a_paper(self, ledger: BudgetLedger) -> None:
        """Crossref filtered non-paper record types and OpenAlex did not, which is a
        fail-OPEN asymmetry: an editorial or a decision letter entering a result set
        becomes a "paper" that can be queued for acquisition, fetched, and mined for
        datasets it cannot contain. Nothing downstream can catch it -- a SearchResult
        carries no record type -- so the adapter is the only place this can be refused.

        Both discriminators are exercised, because a journal that deposits its review
        correspondence as ``article`` betrays it only in the title.
        """
        payload = {
            "results": [
                {"title": "Editorial: combustion chemistry today", "type": "editorial"},
                {"title": "Author response: shock tube ignition delay", "type": "article"},
                {"title": "Decision letter: laminar flame speeds of ammonia", "type": "article"},
                {"title": "Peer review of a kinetic model", "type": "peer-review"},
                {
                    "title": "Ignition delay times of methane in response to strain",
                    "type": "article",
                    "doi": "https://doi.org/10.1016/j.combustflame.2015.11.011",
                },
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("ignition delay")

        # The last one is the control, and it is the one that matters: "in response to"
        # appears mid-title, so a substring test would have eaten a real paper.
        assert [result.title for result in results] == ["Ignition delay times of methane in response to strain"]

    def test_the_filter_does_not_eat_a_paper_that_merely_looks_like_an_artefact(self, ledger: BudgetLedger) -> None:
        """The false-POSITIVE surface, which the first draft of this filter got wrong.

        A drop here is unrecoverable: exactly one search tool is configured per run, and a
        record removed at the adapter never reaches anything that could notice it missing.
        Each of these was in the first draft's denylist and each is a real paper or the
        only correct copy of one:

        * a journal ``Letter`` is a full research paper with data, and OpenAlex does not
          promise that type means correspondence;
        * an erratum or a correction can carry the CORRECTED TABLE -- filtering it means
          mining the stale original forever and never seeing the fix;
        * a retraction notice is how you learn the data is withdrawn.
        """
        payload = {
            "results": [
                {
                    "title": "Rate constants for OH + C2H6 from 800 to 1300 K",
                    "type": "letter",
                    "doi": "https://doi.org/10.1016/j.cplett.2014.01.001",
                },
                {
                    "title": "Erratum: laminar burning velocities of ammonia/hydrogen blends",
                    "type": "erratum",
                    "doi": "https://doi.org/10.1016/j.combustflame.2019.02.002",
                },
                {
                    "title": "Correction to: ignition delay measurements behind reflected shocks",
                    "type": "article",
                    "doi": "https://doi.org/10.1021/acs.energyfuels.2020.03.003",
                },
                {
                    "title": "Corrigendum to a jet-stirred reactor speciation study",
                    "type": "article",
                    "doi": "https://doi.org/10.1016/j.proci.2018.04.004",
                },
                {
                    "title": "Retraction of a kinetic mechanism for syngas oxidation",
                    "type": "retraction",
                    "doi": "https://doi.org/10.1016/j.fuel.2017.05.005",
                },
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("ignition delay")

        assert len(results) == len(payload["results"])

    def test_abstract_inverted_index_is_reconstructed_in_word_order(self, ledger: BudgetLedger) -> None:
        payload = {
            "results": [
                {
                    "title": "A paper",
                    "doi": "https://doi.org/10.1/x",
                    "abstract_inverted_index": {"Ignition": [0], "delay": [1], "measured": [2]},
                    "locations": [],
                }
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        assert tool.search("q")[0].snippet == "Ignition delay measured"

    def test_contact_email_is_sent_only_when_configured(self, ledger: BudgetLedger) -> None:
        seen: list[str] = []
        tool = OpenAlexSearchTool(
            external_provider_consent=True, ledger=ledger, opener=_opener_for({"results": []}, seen)
        )
        tool.search("q")
        assert "mailto=" not in seen[0]

        seen.clear()
        polite = OpenAlexSearchTool(
            external_provider_consent=True,
            ledger=ledger,
            contact_email="a@b.org",
            opener=_opener_for({"results": []}, seen),
        )
        polite.search("q")
        assert "mailto=a%40b.org" in seen[0]

    @pytest.mark.parametrize(
        "payload",
        [{}, {"results": "nope"}, {"results": [None, 3]}, [], "garbage", {"results": [{}]}],
    )
    def test_malformed_payloads_yield_no_results_rather_than_raising(self, ledger: BudgetLedger, payload: Any) -> None:
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))
        assert tool.search("q") == []

    def test_search_is_charged_to_the_ledger(self, ledger: BudgetLedger) -> None:
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for({"results": []}))
        before = ledger.usage().index_lookups
        tool.search("q")
        # Charged to INDEX_LOOKUPS, not FETCHES: a search returns a small JSON result
        # set, and charging it to the document-download ceiling starved real fetches on
        # a live run (see BudgetDimension.INDEX_LOOKUPS).
        assert ledger.usage().index_lookups == before + 1
        assert ledger.usage().fetches == 0


class TestCrossrefSearchTool:
    def test_supplementary_components_are_dropped(self, ledger: BudgetLedger) -> None:
        """A live probe for "ignition delay time shock tube methane" returned the
        .s001/.s002 supplementary files of one article as two extra results carrying
        that article's full title. They are not papers and must not be offered as
        candidates -- DOI dedup cannot catch them because their DOIs really do differ."""
        payload = {
            "message": {
                "items": [
                    {
                        "type": "component",
                        "DOI": "10.1021/acs.energyfuels.0c04277.s001",
                        "title": ["Comparison of Methane Combustion Mechanisms"],
                    },
                    {
                        "type": "component",
                        "DOI": "10.1021/acs.energyfuels.0c04277.s002",
                        "title": ["Comparison of Methane Combustion Mechanisms"],
                    },
                    {
                        "type": "journal-article",
                        "DOI": "10.1021/acs.energyfuels.0c04277",
                        "title": ["Comparison of Methane Combustion Mechanisms"],
                    },
                ]
            }
        }
        tool = CrossrefSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("methane mechanisms")

        assert [r.doi for r in results] == ["10.1021/acs.energyfuels.0c04277"]

    def test_pdf_link_is_extracted_only_from_a_pdf_content_type(self, ledger: BudgetLedger) -> None:
        payload = {
            "message": {
                "items": [
                    {
                        "type": "journal-article",
                        "DOI": "10.1/a",
                        "title": ["Paper A"],
                        "link": [
                            {"URL": "https://x.example/full.xml", "content-type": "application/xml"},
                            {"URL": "https://x.example/full.pdf", "content-type": "application/pdf"},
                        ],
                    }
                ]
            }
        }
        tool = CrossrefSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        assert tool.search("q")[0].pdf_url == "https://x.example/full.pdf"

    def test_never_claims_open_access_since_crossref_does_not_report_it(self, ledger: BudgetLedger) -> None:
        payload = {"message": {"items": [{"type": "journal-article", "DOI": "10.1/a", "title": ["A"]}]}}
        tool = CrossrefSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))
        assert tool.search("q")[0].is_open_access is False

    @pytest.mark.parametrize(
        "payload", [{}, {"message": {}}, {"message": {"items": "no"}}, {"message": {"items": [{}]}}]
    )
    def test_malformed_payloads_yield_no_results_rather_than_raising(self, ledger: BudgetLedger, payload: Any) -> None:
        tool = CrossrefSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))
        assert tool.search("q") == []


class TestDedupeByDoi:
    def test_same_paper_twice_is_collapsed(self) -> None:
        """The probe returned "Modeling nitrogen chemistry in combustion" as two
        separate repository records; each duplicate would spend a second fetch."""
        rows = [
            SearchResult(title="Nitrogen chemistry", url="https://a.example/1", doi="10.1/x"),
            SearchResult(title="Nitrogen chemistry", url="https://b.example/2", doi="10.1/x"),
        ]
        assert len(dedupe_by_doi(rows)) == 1

    def test_duplicate_carrying_full_text_replaces_the_one_without(self) -> None:
        rows = [
            SearchResult(title="P", url="https://a.example/1", doi="10.1/x"),
            SearchResult(title="P", url="https://b.example/2", doi="10.1/x", pdf_url="https://b.example/2.pdf"),
        ]
        kept = dedupe_by_doi(rows)
        assert len(kept) == 1
        assert kept[0].pdf_url == "https://b.example/2.pdf"

    def test_entries_without_a_doi_are_not_merged_together(self) -> None:
        """Two different DOI-less papers must stay distinct: there is nothing reliable
        to merge them on, and merging would silently discard a real candidate."""
        rows = [
            SearchResult(title="First", url="https://a.example/1"),
            SearchResult(title="Second", url="https://b.example/2"),
        ]
        assert len(dedupe_by_doi(rows)) == 2


def _routing_opener(routes: dict[str, Any], seen: list[str] | None = None) -> Any:
    """Build an injected opener routed by URL prefix.

    ``routes`` maps a URL prefix (endpoint) to either a JSON-serializable payload or an
    exception instance to raise, so one opener can serve OpenAlex and Unpaywall
    differently within a single ``resolve`` call.
    """

    def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> _FakeResponse:
        if seen is not None:
            seen.append(url)
        for prefix, payload in routes.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected URL requested: {url}")

    return opener


class TestOpenAlexOaPdfUrls:
    """Pure parsing of one OpenAlex work into ordered OA PDF candidates."""

    def test_best_oa_location_comes_first_then_oa_locations(self) -> None:
        work = {
            "best_oa_location": {"pdf_url": "https://publisher.example/oa.pdf", "is_oa": True},
            "locations": [
                {"pdf_url": "https://repo.example/copy.pdf", "is_oa": True},
                {"pdf_url": "https://publisher.example/oa.pdf", "is_oa": True},
            ],
        }
        assert openalex_oa_pdf_urls(work) == [
            "https://publisher.example/oa.pdf",
            "https://repo.example/copy.pdf",
        ]

    def test_non_oa_locations_are_excluded(self) -> None:
        """A location's pdf_url without ``is_oa: true`` asserts nothing about open
        access; using it would re-introduce the guessed-paywall defect from the other
        direction."""
        work = {
            "locations": [
                {"pdf_url": "https://paywalled.example/full.pdf", "is_oa": False},
                {"pdf_url": "https://unknown.example/full.pdf"},
                {"pdf_url": "https://oa.example/full.pdf", "is_oa": True},
            ]
        }
        assert openalex_oa_pdf_urls(work) == ["https://oa.example/full.pdf"]

    @pytest.mark.parametrize("work", [None, [], "x", {}, {"best_oa_location": "no", "locations": 3}])
    def test_garbage_shapes_yield_no_candidates(self, work: Any) -> None:
        assert openalex_oa_pdf_urls(work) == []


class TestUnpaywallOaPdfUrls:
    """Pure parsing of one Unpaywall record into ordered OA PDF candidates."""

    def test_best_location_first_then_oa_locations_deduplicated(self) -> None:
        payload = {
            "best_oa_location": {"url_for_pdf": "https://pubs.example/vor.pdf"},
            "oa_locations": [
                {"url_for_pdf": "https://pubs.example/vor.pdf"},
                {"url_for_pdf": "https://repo.example/green.pdf"},
                {"url_for_pdf": None},
            ],
        }
        assert unpaywall_oa_pdf_urls(payload) == [
            "https://pubs.example/vor.pdf",
            "https://repo.example/green.pdf",
        ]

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"best_oa_location": 7, "oa_locations": "no"}])
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert unpaywall_oa_pdf_urls(payload) == []


_RESOLVER_DOI = "10.1039/c9re00429g"

_OPENALEX_WORK = {
    "best_oa_location": {"pdf_url": "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g", "is_oa": True},
    "locations": [
        {"pdf_url": "https://repo.example/green.pdf", "is_oa": True},
    ],
}

_UNPAYWALL_RECORD = {
    "best_oa_location": {"url_for_pdf": "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g"},
    "oa_locations": [
        {"url_for_pdf": "https://unpaywall-only.example/copy.pdf"},
    ],
}

#: Well-formed responses that advertise NO open-access copy. Distinct from a provider
#: FAILING: these are honest negatives and must leave a resolution ``complete``.
_OPENALEX_NO_OA: dict[str, Any] = {"best_oa_location": None, "locations": []}
_UNPAYWALL_EMPTY: dict[str, Any] = {"best_oa_location": None, "oa_locations": []}

#: Empty-but-well-formed payloads for the always-consulted providers that these
#: focused tests are not exercising: they advertise nothing, so assertions about
#: OpenAlex/Unpaywall behaviour stay undiluted.
_QUIET_PROVIDER_ROUTES: dict[str, Any] = {
    CROSSREF_ENDPOINT: {},
    SEMANTIC_SCHOLAR_ENDPOINT: {},
    DOAJ_ENDPOINT: {},
    EUROPEPMC_ENDPOINT: {},
}


class TestOpenAccessResolverCompleteness:
    """Whether resolution actually finished -- the exhausted-vs-truncated distinction.

    An empty candidate list means two very different things depending on this flag, and
    conflating them is what let ``no_open_access_copy`` overstate what had been
    established (the same asserted-vs-observed defect already fixed for ``paywalled``).
    """

    def test_a_provider_lookup_failing_in_transit_marks_the_resolution_incomplete(self, ledger: BudgetLedger) -> None:
        """Observed live: an arXiv read timeout in one run.

        A Semantic Scholar HTTP 404 used to be lumped in with this case too, but a
        404 is a normal "no record for this identifier" answer, not a transport
        failure -- see ``test_a_provider_404_does_not_mark_the_resolution_incomplete``
        below and ``SearchNotFound`` in ``carmel.agents.tools.search``."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: TimeoutError("read operation timed out"),
                    UNPAYWALL_ENDPOINT: _UNPAYWALL_EMPTY,
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert resolution.candidates == ()
        assert resolution.coverage is OaLookupCoverage.PARTIAL
        assert "lookup failed" in resolution.note

    def test_a_resolution_where_every_provider_answered_is_complete(self, ledger: BudgetLedger) -> None:
        """The honest negative: all providers consulted, none had a copy."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {OPENALEX_ENDPOINT: _OPENALEX_NO_OA, UNPAYWALL_ENDPOINT: _UNPAYWALL_EMPTY, **_QUIET_PROVIDER_ROUTES}
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert resolution.candidates == ()
        assert resolution.coverage is OaLookupCoverage.COMPLETE

    def test_a_provider_404_does_not_mark_the_resolution_incomplete(self, ledger: BudgetLedger) -> None:
        """A 404 means the provider was reached and answered normally with "no record
        for this identifier" -- e.g. a live run saw Semantic Scholar 404 on
        ``/paper/DOI:10.1115/1.4007737`` (observed 2026-07-30 and 2026-07-31). That
        provider simply contributes nothing; resolution is NOT cut short on its
        account, unlike a genuine transport failure (500/timeout/etc, see the
        transit-failure test above)."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: _OPENALEX_NO_OA,
                    UNPAYWALL_ENDPOINT: urllib.error.HTTPError(UNPAYWALL_ENDPOINT, 404, "Not Found", None, None),
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert resolution.candidates == ()
        assert resolution.coverage is OaLookupCoverage.COMPLETE
        assert "Unpaywall: no record" in resolution.note

    def test_a_provider_500_still_marks_the_resolution_incomplete(self, ledger: BudgetLedger) -> None:
        """A 500 (or any non-404 transport failure) is genuinely ambiguous -- the
        provider was NOT reached and answered, so it must keep marking the
        resolution incomplete exactly as before this fix."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: _OPENALEX_NO_OA,
                    UNPAYWALL_ENDPOINT: urllib.error.HTTPError(
                        UNPAYWALL_ENDPOINT, 500, "Internal Server Error", None, None
                    ),
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert resolution.candidates == ()
        assert resolution.coverage is OaLookupCoverage.PARTIAL
        assert "lookup failed" in resolution.note

    def test_404_and_transient_failure_produce_visibly_different_notes(self, ledger: BudgetLedger) -> None:
        """The operator-facing note must distinguish "no record" from "lookup
        failed (...)" -- they mean very different things about what is known."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: TimeoutError("read operation timed out"),
                    UNPAYWALL_ENDPOINT: urllib.error.HTTPError(UNPAYWALL_ENDPOINT, 404, "Not Found", None, None),
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert "OpenAlex: lookup failed" in resolution.note
        assert "Unpaywall: no record" in resolution.note

    def test_a_provider_declining_for_a_missing_api_key_does_not_mark_it_incomplete(self, ledger: BudgetLedger) -> None:
        """A missing optional key is a stable configuration fact, not an unknown.

        CORE ships keyless by default, so counting its skip as "incomplete" would mark
        every paper incomplete and make the distinction worthless.
        """
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            core_api_key=None,
            opener=_routing_opener(
                {OPENALEX_ENDPOINT: _OPENALEX_NO_OA, UNPAYWALL_ENDPOINT: _UNPAYWALL_EMPTY, **_QUIET_PROVIDER_ROUTES}
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert resolution.coverage is OaLookupCoverage.COMPLETE
        assert "CORE" in resolution.note


class TestOpenAccessResolver:
    """Deterministic (non-LLM) OA resolution keyed on DOI."""

    def test_candidates_merge_openalex_first_then_unpaywall_deduplicated(self, ledger: BudgetLedger) -> None:
        seen: list[str] = []
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            contact_email="ops@example.org",
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {OPENALEX_ENDPOINT: _OPENALEX_WORK, UNPAYWALL_ENDPOINT: _UNPAYWALL_RECORD, **_QUIET_PROVIDER_ROUTES},
                seen,
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert list(resolution.candidates) == [
            "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g",
            "https://repo.example/green.pdf",
            "https://unpaywall-only.example/copy.pdf",
        ]
        openalex_urls = [u for u in seen if u.startswith(OPENALEX_ENDPOINT)]
        unpaywall_urls = [u for u in seen if u.startswith(UNPAYWALL_ENDPOINT)]
        assert len(openalex_urls) == 1 and len(unpaywall_urls) == 1
        assert _RESOLVER_DOI in openalex_urls[0]
        assert "mailto=" in openalex_urls[0]
        assert _RESOLVER_DOI in unpaywall_urls[0]
        assert "email=" in unpaywall_urls[0]

    def test_unpaywall_is_skipped_cleanly_when_no_email_is_configured(
        self, ledger: BudgetLedger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unpaywall REQUIRES a real contact email; a fake or placeholder address would
        violate its API terms, so no email means no Unpaywall call at all -- with one
        clear warning, not one per paper."""
        seen: list[str] = []
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            opener=_routing_opener({OPENALEX_ENDPOINT: _OPENALEX_WORK, **_QUIET_PROVIDER_ROUTES}, seen),
        )

        with caplog.at_level("WARNING", logger="carmel.agents.tools.academic"):
            first = resolver.resolve(_RESOLVER_DOI)
            second = resolver.resolve(_RESOLVER_DOI)

        assert not any(u.startswith(UNPAYWALL_ENDPOINT) for u in seen)
        assert list(first.candidates) == list(second.candidates) != []
        assert "unpaywall" in first.note.lower()
        skip_warnings = [r for r in caplog.records if "unpaywall" in r.getMessage().lower()]
        assert len(skip_warnings) == 1

    def test_consent_withheld_means_zero_network_calls(self, ledger: BudgetLedger) -> None:
        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> Any:
            raise AssertionError(f"network call attempted without consent: {url}")

        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=False,
            unpaywall_email="ops@example.org",
            opener=opener,
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert list(resolution.candidates) == []
        assert "consent" in resolution.note.lower()

    def test_one_backend_failing_does_not_lose_the_other(self, ledger: BudgetLedger) -> None:
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: OSError("connection refused"),
                    UNPAYWALL_ENDPOINT: _UNPAYWALL_RECORD,
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert list(resolution.candidates) == [
            "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g",
            "https://unpaywall-only.example/copy.pdf",
        ]
        assert "openalex" in resolution.note.lower()
        assert "failed" in resolution.note.lower()

    def test_a_persistently_failing_provider_warns_only_once_per_resolver_instance(
        self, ledger: BudgetLedger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A provider whose endpoint is simply dead (connection error, timeout,
        non-200 -- not a credential skip) must warn at most once per resolver
        instance, not once per paper resolved."""
        resolver = OpenAccessResolver(
            ledger=ledger,
            external_provider_consent=True,
            unpaywall_email="ops@example.org",
            opener=_routing_opener(
                {
                    OPENALEX_ENDPOINT: OSError("connection refused"),
                    UNPAYWALL_ENDPOINT: _UNPAYWALL_RECORD,
                    **_QUIET_PROVIDER_ROUTES,
                }
            ),
        )

        with caplog.at_level("WARNING", logger="carmel.agents.tools.academic"):
            resolver.resolve(_RESOLVER_DOI)
            resolver.resolve(_RESOLVER_DOI)
            resolver.resolve(_RESOLVER_DOI)

        failure_warnings = [r for r in caplog.records if "openalex" in r.getMessage().lower()]
        assert len(failure_warnings) == 1


class TestCrossrefTdmCandidates:
    """Publisher-sanctioned Crossref ``link[]`` full-text entries (TDM links)."""

    def test_only_pdf_links_with_tdm_intended_application_are_kept(self) -> None:
        payload = {
            "message": {
                "link": [
                    {
                        "URL": "https://publisher.example/tdm.pdf",
                        "content-type": "application/pdf",
                        "intended-application": "text-mining",
                    },
                    {
                        "URL": "https://publisher.example/similarity.pdf",
                        "content-type": "application/pdf",
                        "intended-application": "similarity-checking",
                    },
                    {
                        "URL": "https://publisher.example/tdm.pdf",
                        "content-type": "application/pdf",
                        "intended-application": "similarity-checking",
                    },
                    {
                        "URL": "https://publisher.example/landing.pdf",
                        "content-type": "application/pdf",
                        "intended-application": "unspecified",
                    },
                    {
                        "URL": "https://publisher.example/tdm.xml",
                        "content-type": "text/xml",
                        "intended-application": "text-mining",
                    },
                ]
            }
        }
        assert crossref_tdm_candidates(payload) == [
            OaCandidate(url="https://publisher.example/tdm.pdf", tier=OaTier.PUBLISHER),
            OaCandidate(url="https://publisher.example/similarity.pdf", tier=OaTier.PUBLISHER),
        ]

    @pytest.mark.parametrize(
        "payload", [None, [], "x", {}, {"message": []}, {"message": {"link": "no"}}, {"message": {"link": [7]}}]
    )
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert crossref_tdm_candidates(payload) == []


class TestSemanticScholarCandidates:
    """One ``openAccessPdf`` per Semantic Scholar paper record."""

    def test_green_status_ranks_as_repository_copy(self) -> None:
        payload = {"openAccessPdf": {"url": "https://repo.example/copy.pdf", "status": "GREEN"}}
        assert semantic_scholar_oa_candidates(payload) == [
            OaCandidate(url="https://repo.example/copy.pdf", tier=OaTier.REPOSITORY)
        ]

    @pytest.mark.parametrize("status", ["GOLD", "HYBRID", "BRONZE", "gold"])
    def test_publisher_hosted_statuses_rank_as_publisher_copy(self, status: str) -> None:
        payload = {"openAccessPdf": {"url": "https://pubs.example/vor.pdf", "status": status}}
        assert semantic_scholar_oa_candidates(payload) == [
            OaCandidate(url="https://pubs.example/vor.pdf", tier=OaTier.PUBLISHER)
        ]

    def test_missing_status_defaults_to_repository_not_publisher(self) -> None:
        """Claiming a publisher copy the index never asserted would over-rank it."""
        payload = {"openAccessPdf": {"url": "https://somewhere.example/copy.pdf"}}
        assert semantic_scholar_oa_candidates(payload) == [
            OaCandidate(url="https://somewhere.example/copy.pdf", tier=OaTier.REPOSITORY)
        ]

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"openAccessPdf": None}, {"openAccessPdf": {"url": ""}}])
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert semantic_scholar_oa_candidates(payload) == []


class TestCoreCandidates:
    """CORE v3 search results, accepted only when the work names the wanted DOI."""

    def test_matching_work_yields_download_urls_deduplicated(self) -> None:
        payload = {
            "results": [
                {
                    "doi": f"https://doi.org/{_RESOLVER_DOI}",
                    "downloadUrl": "https://core.example/download.pdf",
                    "links": [
                        {"type": "download", "url": "https://core.example/download.pdf"},
                        {"type": "download", "url": "https://repo.example/other.pdf"},
                        {"type": "reader", "url": "https://core.example/reader"},
                    ],
                    "sourceFulltextUrls": ["https://university.example/thesis.pdf"],
                }
            ]
        }
        assert core_oa_candidates(payload, doi=_RESOLVER_DOI) == [
            OaCandidate(url="https://core.example/download.pdf", tier=OaTier.REPOSITORY),
            OaCandidate(url="https://repo.example/other.pdf", tier=OaTier.REPOSITORY),
            OaCandidate(url="https://university.example/thesis.pdf", tier=OaTier.REPOSITORY),
        ]

    def test_works_not_naming_the_wanted_doi_are_excluded(self) -> None:
        """CORE's search is fuzzy; a hit that names a different DOI (or none at all)
        is a wrong-paper risk, which is far worse than a miss."""
        payload = {
            "results": [
                {"doi": "10.9999/other", "downloadUrl": "https://core.example/wrong.pdf"},
                {"downloadUrl": "https://core.example/anonymous.pdf"},
            ]
        }
        assert core_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"results": "no"}, {"results": [7]}])
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert core_oa_candidates(payload, doi=_RESOLVER_DOI) == []


class TestDoajCandidates:
    """DOAJ article search results: fulltext links from gold-OA journals."""

    def test_fulltext_links_with_pdf_ordered_first(self) -> None:
        payload = {
            "results": [
                {
                    "bibjson": {
                        "identifier": [{"type": "doi", "id": _RESOLVER_DOI.upper()}],
                        "link": [
                            {"type": "fulltext", "url": "https://journal.example/article", "content_type": "HTML"},
                            {"type": "fulltext", "url": "https://journal.example/article.pdf", "content_type": "PDF"},
                            {"type": "homepage", "url": "https://journal.example/"},
                        ],
                    }
                }
            ]
        }
        assert doaj_oa_candidates(payload, doi=_RESOLVER_DOI) == [
            OaCandidate(url="https://journal.example/article.pdf", tier=OaTier.PUBLISHER),
            OaCandidate(url="https://journal.example/article", tier=OaTier.PUBLISHER),
        ]

    def test_results_not_naming_the_wanted_doi_are_excluded(self) -> None:
        payload = {
            "results": [
                {
                    "bibjson": {
                        "identifier": [{"type": "doi", "id": "10.9999/other"}],
                        "link": [{"type": "fulltext", "url": "https://journal.example/wrong.pdf"}],
                    }
                },
                {
                    "bibjson": {
                        "link": [{"type": "fulltext", "url": "https://journal.example/anonymous.pdf"}],
                    }
                },
            ]
        }
        assert doaj_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"results": "no"}, {"results": [7]}])
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert doaj_oa_candidates(payload, doi=_RESOLVER_DOI) == []


class TestEuropepmcCandidates:
    """Europe PMC ``resultType=core`` search results, PDF-only and DOI-gated."""

    def test_pdf_full_text_url_for_the_matching_doi_yields_a_candidate(self) -> None:
        payload = {
            "resultList": {
                "result": [
                    {
                        "doi": _RESOLVER_DOI,
                        "fullTextUrlList": {
                            "fullTextUrl": [{"documentStyle": "pdf", "url": "https://europepmc.example/fulltext.pdf"}]
                        },
                    }
                ]
            }
        }
        assert europepmc_oa_candidates(payload, doi=_RESOLVER_DOI) == [
            OaCandidate(url="https://europepmc.example/fulltext.pdf", tier=OaTier.REPOSITORY)
        ]

    def test_results_not_naming_the_wanted_doi_are_excluded(self) -> None:
        """Europe PMC's ``query=DOI:"..."`` search can still surface a citing or
        related work; a near-miss DOI must not be treated as the wanted paper."""
        payload = {
            "resultList": {
                "result": [
                    {
                        "doi": "10.9999/other",
                        "fullTextUrlList": {
                            "fullTextUrl": [{"documentStyle": "pdf", "url": "https://europepmc.example/wrong.pdf"}]
                        },
                    },
                    {
                        "fullTextUrlList": {
                            "fullTextUrl": [{"documentStyle": "pdf", "url": "https://europepmc.example/anonymous.pdf"}]
                        },
                    },
                ]
            }
        }
        assert europepmc_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    def test_empty_result_list_yields_no_candidates(self) -> None:
        payload = {"resultList": {"result": []}}
        assert europepmc_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    def test_html_only_full_text_url_is_not_offered_as_a_pdf_candidate(self) -> None:
        """The acquisition layer's ``_validate_document`` rejects ``text/html``
        outright, so advertising an HTML-only URL as a candidate would just burn a
        fetch on a guaranteed non-document outcome."""
        payload = {
            "resultList": {
                "result": [
                    {
                        "doi": _RESOLVER_DOI,
                        "fullTextUrlList": {
                            "fullTextUrl": [{"documentStyle": "html", "url": "https://europepmc.example/article"}]
                        },
                    }
                ]
            }
        }
        assert europepmc_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            "x",
            {},
            {"resultList": "no"},
            {"resultList": {"result": "no"}},
            {"resultList": {"result": [7]}},
            {"resultList": {"result": [{"doi": _RESOLVER_DOI, "fullTextUrlList": "no"}]}},
            {"resultList": {"result": [{"doi": _RESOLVER_DOI, "fullTextUrlList": {"fullTextUrl": [7]}}]}},
        ],
    )
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert europepmc_oa_candidates(payload, doi=_RESOLVER_DOI) == []


class TestElsevierCandidates:
    """Elsevier Article Retrieval responses, DOI-gated (see ``elsevier_oa_candidates``'s
    ``UNVERIFIED SHAPE`` docstring: no live entitled response has been observed yet, so
    this fixture follows the published ``full-text-retrieval-response.coredata`` schema)."""

    def test_matching_doi_yields_a_publisher_candidate(self) -> None:
        payload = {"full-text-retrieval-response": {"coredata": {"prism:doi": _RESOLVER_DOI}}}
        assert elsevier_oa_candidates(payload, doi=_RESOLVER_DOI) == [
            OaCandidate(url=f"{ELSEVIER_ARTICLE_ENDPOINT}/{_RESOLVER_DOI}", tier=OaTier.PUBLISHER)
        ]

    def test_mismatched_doi_is_excluded(self) -> None:
        """A DOI-keyed request should always echo the same DOI back, but the parser
        must not just trust the request path -- it names the wanted DOI explicitly."""
        payload = {"full-text-retrieval-response": {"coredata": {"prism:doi": "10.9999/other"}}}
        assert elsevier_oa_candidates(payload, doi=_RESOLVER_DOI) == []

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            "x",
            {},
            {"full-text-retrieval-response": "no"},
            {"full-text-retrieval-response": {"coredata": "no"}},
            {"full-text-retrieval-response": {"coredata": {"prism:doi": None}}},
            {"full-text-retrieval-response": {"coredata": {}}},
        ],
    )
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert elsevier_oa_candidates(payload, doi=_RESOLVER_DOI) == []


_CHEMRXIV_ASSET = {"original": {"url": "https://chemrxiv.example/item/original/preprint.pdf"}}
_WANTED_TITLE = "Ammonia Combustion Kinetics: A Study"


class TestChemrxivCandidates:
    """ChemRxiv term-search hits, accepted only on a strong DOI/title match."""

    def test_match_by_preprint_doi(self) -> None:
        payload = {"itemHits": [{"item": {"doi": _RESOLVER_DOI, "title": "Different", "asset": _CHEMRXIV_ASSET}}]}
        assert chemrxiv_oa_candidates(payload, doi=_RESOLVER_DOI, title=None) == [
            OaCandidate(url="https://chemrxiv.example/item/original/preprint.pdf", tier=OaTier.PREPRINT)
        ]

    def test_match_by_vor_doi(self) -> None:
        payload = {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2020-abcde",
                        "title": "Different",
                        "vor": {"vorDoi": f"https://doi.org/{_RESOLVER_DOI}"},
                        "asset": _CHEMRXIV_ASSET,
                    }
                }
            ]
        }
        assert chemrxiv_oa_candidates(payload, doi=_RESOLVER_DOI, title=None) == [
            OaCandidate(url="https://chemrxiv.example/item/original/preprint.pdf", tier=OaTier.PREPRINT)
        ]

    def test_match_by_exact_title_ignoring_case_and_punctuation(self) -> None:
        payload = {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2020-abcde",
                        "title": "ammonia combustion kinetics -- a study",
                        "asset": _CHEMRXIV_ASSET,
                    }
                }
            ]
        }
        assert chemrxiv_oa_candidates(payload, doi=_RESOLVER_DOI, title=_WANTED_TITLE) == [
            OaCandidate(url="https://chemrxiv.example/item/original/preprint.pdf", tier=OaTier.PREPRINT)
        ]

    def test_weak_matches_yield_nothing(self) -> None:
        """A term search returns topically-similar preprints; without an exact DOI or
        title match, emitting a candidate would fetch the WRONG paper."""
        payload = {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2020-abcde",
                        "title": "Ammonia combustion kinetics in gas turbines",
                        "asset": _CHEMRXIV_ASSET,
                    }
                }
            ]
        }
        assert chemrxiv_oa_candidates(payload, doi=_RESOLVER_DOI, title=_WANTED_TITLE) == []

    @pytest.mark.parametrize("payload", [None, [], "x", {}, {"itemHits": "no"}, {"itemHits": [7]}, {"itemHits": [{}]}])
    def test_garbage_shapes_yield_no_candidates(self, payload: Any) -> None:
        assert chemrxiv_oa_candidates(payload, doi=_RESOLVER_DOI, title=_WANTED_TITLE) == []


_ARXIV_DOI_TAG = f'<arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">{_RESOLVER_DOI}</arxiv:doi>'

_ARXIV_ATOM = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1405.5500v3</id>
    <title>Ammonia combustion kinetics — a study</title>
    {_ARXIV_DOI_TAG}
    <link href="https://arxiv.org/abs/1405.5500v3" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/1405.5500v3" rel="related" type="application/pdf" title="pdf"/>
  </entry>
</feed>
"""

_ARXIV_PDF = OaCandidate(url="https://arxiv.org/pdf/1405.5500v3", tier=OaTier.PREPRINT)


class TestArxivCandidates:
    """arXiv Atom entries, accepted only on a strong DOI/title match."""

    def test_match_by_arxiv_doi(self) -> None:
        assert arxiv_oa_candidates(_ARXIV_ATOM, doi=_RESOLVER_DOI, title=None) == [_ARXIV_PDF]

    def test_match_by_exact_title(self) -> None:
        atom = _ARXIV_ATOM.replace(_ARXIV_DOI_TAG, "")
        assert arxiv_oa_candidates(atom, doi=_RESOLVER_DOI, title=_WANTED_TITLE) == [_ARXIV_PDF]

    def test_weak_matches_yield_nothing(self) -> None:
        atom = _ARXIV_ATOM.replace(_ARXIV_DOI_TAG, "")
        wrong = "Ammonia combustion kinetics in gas turbines"
        assert arxiv_oa_candidates(atom, doi=_RESOLVER_DOI, title=wrong) == []

    @pytest.mark.parametrize("text", ["", "not xml <at all", '{"json": true}', "<feed/>"])
    def test_malformed_or_empty_feeds_yield_no_candidates(self, text: str) -> None:
        assert arxiv_oa_candidates(text, doi=_RESOLVER_DOI, title=_WANTED_TITLE) == []


def _full_provider_routes() -> dict[str, Any]:
    """One advertised candidate per provider, with deliberate cross-provider dupes."""
    return {
        OPENALEX_ENDPOINT: {
            "best_oa_location": {"pdf_url": "https://publisher.example/vor.pdf"},
            "locations": [{"pdf_url": "https://repo.example/mirror.pdf", "is_oa": True}],
        },
        UNPAYWALL_ENDPOINT: {
            "best_oa_location": {"url_for_pdf": "https://publisher.example/vor.pdf"},
            "oa_locations": [{"url_for_pdf": "https://green.example/copy.pdf"}],
        },
        CROSSREF_ENDPOINT: {
            "message": {
                "link": [
                    {
                        "URL": "https://publisher.example/tdm.pdf",
                        "content-type": "application/pdf",
                        "intended-application": "text-mining",
                    }
                ]
            }
        },
        SEMANTIC_SCHOLAR_ENDPOINT: {"openAccessPdf": {"url": "https://repo.example/mirror.pdf", "status": "GREEN"}},
        CORE_ENDPOINT: {"results": [{"doi": _RESOLVER_DOI, "downloadUrl": "https://core.example/download.pdf"}]},
        DOAJ_ENDPOINT: {
            "results": [
                {
                    "bibjson": {
                        "identifier": [{"type": "doi", "id": _RESOLVER_DOI}],
                        "link": [
                            {"type": "fulltext", "url": "https://journal.example/article.pdf", "content_type": "PDF"}
                        ],
                    }
                }
            ]
        },
        EUROPEPMC_ENDPOINT: {
            "resultList": {
                "result": [
                    {
                        "doi": _RESOLVER_DOI,
                        "fullTextUrlList": {
                            "fullTextUrl": [{"documentStyle": "pdf", "url": "https://europepmc.example/fulltext.pdf"}]
                        },
                    }
                ]
            }
        },
        CHEMRXIV_ENDPOINT: {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2020-abcde",
                        "title": "Different",
                        "vor": {"vorDoi": _RESOLVER_DOI},
                        "asset": _CHEMRXIV_ASSET,
                    }
                }
            ]
        },
        ARXIV_ENDPOINT: _ARXIV_ATOM,
    }


def _full_resolver(
    ledger: BudgetLedger,
    routes: dict[str, Any],
    seen: list[str] | None = None,
    **overrides: Any,
) -> OpenAccessResolver:
    kwargs: dict[str, Any] = {
        "ledger": ledger,
        "external_provider_consent": True,
        "unpaywall_email": "ops@example.org",
        "core_api_key": "sekret-core-key",
        "semantic_scholar_api_key": "sekret-s2-key",
        "opener": _routing_opener(routes, seen),
    }
    kwargs.update(overrides)
    return OpenAccessResolver(**kwargs)


class TestPluggableProviderResolution:
    """The provider list as a whole: ordering, dedupe, cap, keys, fail-softness."""

    def test_candidates_are_ordered_publisher_then_repository_then_preprint(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ChemRxiv is disabled by default (see DEFAULT_ENABLED_PROVIDERS); re-enable
        # it here since this test exercises the full pluggable-provider mechanism,
        # including preprint ordering, not just the production default set.
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"ChemRxiv"})
        seen: list[str] = []
        resolver = _full_resolver(ledger, _full_provider_routes(), seen)

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert list(resolution.candidates) == [
            # publisher copies (version of record) first ...
            "https://publisher.example/vor.pdf",
            "https://publisher.example/tdm.pdf",
            "https://journal.example/article.pdf",
            # ... then repository/archive copies (mirror.pdf deduplicated across
            # OpenAlex and Semantic Scholar; vor.pdf across OpenAlex and Unpaywall) ...
            "https://repo.example/mirror.pdf",
            "https://green.example/copy.pdf",
            "https://core.example/download.pdf",
            "https://europepmc.example/fulltext.pdf",
            # ... and preprints strictly last: they are not the version of record.
            "https://chemrxiv.example/item/original/preprint.pdf",
            "https://arxiv.org/pdf/1405.5500v3",
        ]
        assert len(seen) == 9, "expected exactly one lookup call per provider"
        assert "preprint, not the version of record" in resolution.note

    def test_one_provider_rate_limiting_does_not_suppress_the_others(self, ledger: BudgetLedger) -> None:
        """A Semantic Scholar 429 must cost exactly one attempt (no retry storm) and
        must not lose any other provider's candidates."""
        routes = _full_provider_routes()
        routes[SEMANTIC_SCHOLAR_ENDPOINT] = OSError("HTTP Error 429: Too Many Requests")
        seen: list[str] = []
        resolver = _full_resolver(ledger, routes, seen)

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert "https://publisher.example/vor.pdf" in resolution.candidates
        assert "https://arxiv.org/pdf/1405.5500v3" in resolution.candidates
        assert sum(u.startswith(SEMANTIC_SCHOLAR_ENDPOINT) for u in seen) == 1
        assert "Semantic Scholar: lookup failed" in resolution.note

    def test_missing_core_key_skips_core_with_one_warning_and_zero_calls(
        self, ledger: BudgetLedger, caplog: pytest.LogCaptureFixture
    ) -> None:
        routes = _full_provider_routes()
        del routes[CORE_ENDPOINT]
        seen: list[str] = []
        resolver = _full_resolver(ledger, routes, seen, core_api_key=None)

        with caplog.at_level("WARNING", logger="carmel.agents.tools.academic"):
            first = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)
            resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert not any(u.startswith(CORE_ENDPOINT) for u in seen)
        assert "CORE: skipped (no API key configured)" in first.note
        core_warnings = [r for r in caplog.records if "CARMEL_CORE_API_KEY" in r.getMessage()]
        assert len(core_warnings) == 1

    def test_api_keys_travel_in_headers_and_are_never_logged(
        self, ledger: BudgetLedger, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Elsevier is disabled by default (see DEFAULT_ENABLED_PROVIDERS); re-enable
        # it here since this test's whole point is that every key-bearing provider's
        # secret stays out of logs/URLs/notes, and Elsevier is one of only two
        # key-gated providers (with CORE).
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"Elsevier"})
        routes = _full_provider_routes()
        routes[ELSEVIER_ARTICLE_ENDPOINT] = {"full-text-retrieval-response": {"coredata": {"prism:doi": _RESOLVER_DOI}}}
        headers_by_url: dict[str, dict[str, str]] = {}
        inner = _routing_opener(routes)

        def opener(url: str, *, headers: dict[str, str], timeout_s: float) -> Any:
            headers_by_url[url] = dict(headers)
            return inner(url, headers=headers, timeout_s=timeout_s)

        resolver = _full_resolver(
            ledger,
            routes,
            opener=opener,
            elsevier_api_key="sekret-elsevier-key",
            elsevier_insttoken="sekret-elsevier-insttoken",
        )

        with caplog.at_level("DEBUG"):
            resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        core_headers = [h for u, h in headers_by_url.items() if u.startswith(CORE_ENDPOINT)]
        s2_headers = [h for u, h in headers_by_url.items() if u.startswith(SEMANTIC_SCHOLAR_ENDPOINT)]
        elsevier_headers = [h for u, h in headers_by_url.items() if u.startswith(ELSEVIER_ARTICLE_ENDPOINT)]
        assert [h.get("Authorization") for h in core_headers] == ["Bearer sekret-core-key"]
        assert [h.get("x-api-key") for h in s2_headers] == ["sekret-s2-key"]
        assert [h.get("X-ELS-APIKey") for h in elsevier_headers] == ["sekret-elsevier-key"]
        assert [h.get("X-ELS-Insttoken") for h in elsevier_headers] == ["sekret-elsevier-insttoken"]
        for secret in ("sekret-core-key", "sekret-s2-key", "sekret-elsevier-key", "sekret-elsevier-insttoken"):
            assert secret not in resolution.note
            assert all(secret not in url for url in headers_by_url)
            assert all(secret not in r.getMessage() for r in caplog.records)

    def test_without_a_title_the_title_matched_providers_are_skipped(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ChemRxiv and arXiv can only be matched safely against a title; with no
        title there is nothing to match, so they must not even be queried."""
        # ChemRxiv is disabled by default (see DEFAULT_ENABLED_PROVIDERS); re-enable
        # it here so the title-matched-provider-skip behavior is still exercised for
        # ChemRxiv specifically, in addition to arXiv (which is enabled by default).
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"ChemRxiv"})
        routes = _full_provider_routes()
        del routes[CHEMRXIV_ENDPOINT]
        del routes[ARXIV_ENDPOINT]
        seen: list[str] = []
        resolver = _full_resolver(ledger, routes, seen)

        resolution = resolver.resolve(_RESOLVER_DOI)

        assert not any(u.startswith(CHEMRXIV_ENDPOINT) for u in seen)
        assert not any(u.startswith(ARXIV_ENDPOINT) for u in seen)
        assert "ChemRxiv: skipped" in resolution.note
        assert "arXiv: skipped" in resolution.note
        assert not any(c.endswith("preprint.pdf") for c in resolution.candidates)

    def test_per_paper_lookup_cap_bounds_the_fan_out(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(academic, "MAX_OA_LOOKUP_CALLS_PER_PAPER", 2)
        seen: list[str] = []
        resolver = _full_resolver(ledger, _full_provider_routes(), seen)

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert len(seen) == 2, "the cap must stop further lookup calls, not candidates already found"
        assert "per-paper lookup cap" in resolution.note
        assert list(resolution.candidates) == [
            "https://publisher.example/vor.pdf",
            "https://repo.example/mirror.pdf",
            "https://green.example/copy.pdf",
        ]


class TestElsevierResolver:
    """Resolver-level behavior specific to Elsevier: default-disabled, key-gated,
    and (load-bearing) that a 403 marks the resolution incomplete rather than "no
    open-access copy" -- see ``_lookup_elsevier``'s and ``elsevier_oa_candidates``'s
    ``UNVERIFIED SHAPE`` notes for why this is the only failure mode observed live."""

    def test_missing_elsevier_key_skips_it_with_one_warning_and_zero_calls(
        self, ledger: BudgetLedger, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"Elsevier"})
        seen: list[str] = []
        resolver = _full_resolver(ledger, _full_provider_routes(), seen)

        with caplog.at_level("WARNING", logger="carmel.agents.tools.academic"):
            first = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)
            resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert not any(u.startswith(ELSEVIER_ARTICLE_ENDPOINT) for u in seen)
        assert first.coverage is OaLookupCoverage.COMPLETE
        assert "Elsevier: skipped (no API key configured)" in first.note
        elsevier_warnings = [r for r in caplog.records if "CARMEL_ELSEVIER_API_KEY" in r.getMessage()]
        assert len(elsevier_warnings) == 1

    def test_entitled_response_yields_a_publisher_candidate(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"Elsevier"})
        routes = _full_provider_routes()
        routes[ELSEVIER_ARTICLE_ENDPOINT] = {"full-text-retrieval-response": {"coredata": {"prism:doi": _RESOLVER_DOI}}}
        resolver = _full_resolver(ledger, routes, elsevier_api_key="sekret-elsevier-key")

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert f"{ELSEVIER_ARTICLE_ENDPOINT}/{_RESOLVER_DOI}" in resolution.candidates
        assert resolution.coverage is OaLookupCoverage.COMPLETE

    def test_403_authentication_error_marks_incomplete_and_names_the_remedy(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measured live failure mode (see module docstrings): a valid key from
        an academic IP still gets 403 AUTHENTICATION_ERROR without institutional
        entitlement association. This must mark the resolution incomplete -- never
        silently "no open-access copy" -- and must name the actual remedy (an
        insttoken or entitlement association), since no code change fixes it."""
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"Elsevier"})
        routes = _full_provider_routes()
        routes[ELSEVIER_ARTICLE_ENDPOINT] = Exception(
            "HTTP Error 403: Forbidden - AUTHENTICATION_ERROR: Requestor configuration "
            "settings insufficient for access to this resource"
        )
        resolver = _full_resolver(ledger, routes, elsevier_api_key="sekret-elsevier-key")

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert not any(c.startswith(ELSEVIER_ARTICLE_ENDPOINT) for c in resolution.candidates)
        assert resolution.coverage is OaLookupCoverage.PARTIAL
        assert "institutional entitlement" in resolution.note
        assert "CARMEL_ELSEVIER_INSTTOKEN" in resolution.note

    def test_malformed_elsevier_payload_yields_no_candidate_and_does_not_crash(
        self, ledger: BudgetLedger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(academic, "DEFAULT_ENABLED_PROVIDERS", academic.DEFAULT_ENABLED_PROVIDERS | {"Elsevier"})
        routes = _full_provider_routes()
        routes[ELSEVIER_ARTICLE_ENDPOINT] = {"unexpected": "shape"}
        resolver = _full_resolver(ledger, routes, elsevier_api_key="sekret-elsevier-key")

        resolution = resolver.resolve(_RESOLVER_DOI, title=_WANTED_TITLE)

        assert not any(c.startswith(ELSEVIER_ARTICLE_ENDPOINT) for c in resolution.candidates)
        assert resolution.coverage is OaLookupCoverage.COMPLETE


class TestOpenAlexBestOaLocationFallback:
    """The real-run defect's search-side half: ``best_oa_location`` is a distinct
    top-level field that ``locations[]`` parsing was silently ignoring, and pdf_urls on
    hosts typed neither 'repository' nor 'publisher' were dropped outright -- so a
    genuinely fetchable OA PDF could be shown to the agent as FULL TEXT: no."""

    def test_best_oa_location_pdf_is_used_when_locations_offer_none(self, ledger: BudgetLedger) -> None:
        payload = {
            "results": [
                {
                    "title": "Catalytic conditions in a packed bed reactor",
                    "doi": "https://doi.org/10.1039/c9re00429g",
                    "open_access": {"is_oa": True},
                    "best_oa_location": {
                        "pdf_url": "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g",
                        "is_oa": True,
                        "source": {"display_name": "Reaction Chemistry & Engineering"},
                    },
                    "locations": [{"host_type": "publisher", "source": {"display_name": "RSC"}}],
                }
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("packed bed reactor")

        assert len(results) == 1
        assert results[0].pdf_url == "https://pubs.rsc.org/en/content/articlepdf/2020/re/c9re00429g"

    def test_a_pdf_on_an_unranked_host_type_is_kept_as_a_last_resort(self, ledger: BudgetLedger) -> None:
        payload = {
            "results": [
                {
                    "title": "A journal-hosted open access paper",
                    "doi": "https://doi.org/10.1/x",
                    "open_access": {"is_oa": True},
                    "locations": [
                        {
                            "pdf_url": "https://journal.example/oa.pdf",
                            "source": {"type": "journal", "display_name": "Journal"},
                        }
                    ],
                }
            ]
        }
        tool = OpenAlexSearchTool(external_provider_consent=True, ledger=ledger, opener=_opener_for(payload))

        results = tool.search("q")

        assert len(results) == 1
        assert results[0].pdf_url == "https://journal.example/oa.pdf"
