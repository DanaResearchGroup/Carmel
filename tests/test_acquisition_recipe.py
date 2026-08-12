# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure publisher-recipe builder behind the acquisition README.

The load-bearing properties: an unknown DOI prefix (or a request with no DOI at all)
must fall through to the genuinely actionable generic recipe rather than raising;
nothing in v1 may claim a verified click-path, because a stale or invented click-path
is worse than none -- it can walk the operator into the browser's "Save Page As",
which the ingestion layer refuses outright; and a registry entry past its shelf life
must degrade to the generic instructions AND say so, rather than keep asserting
guidance nobody has re-checked.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from carmel.schemas.acquisition import AcquisitionReason, AcquisitionRequest
from carmel.services.acquisition import slug_for
from carmel.services.acquisition_recipe import (
    GENERIC_FORMATS,
    PUBLISHER_RECIPES,
    build_recipe,
    doi_prefix,
)

TITLE = "Shock tube study of ignition delay times in methane oxygen argon mixtures"
ELSEVIER_DOI = "10.1016/0010-2180(76)90042-0"

#: A day on which every shipped registry entry is fresh. Pinned, never date.today():
#: the registry's entries EXPIRE by design, so a test that asked "is Elsevier guidance
#: rendered?" against the wall clock would start failing the day the entry goes stale.
FRESH_DAY = date(2026, 8, 1)


def _request(
    *,
    doi: str | None = ELSEVIER_DOI,
    title: str = TITLE,
    landing_url: str | None = None,
) -> AcquisitionRequest:
    if landing_url is None:
        # Matches how literature.py builds it: the DOI resolver when a DOI is known.
        landing_url = f"https://doi.org/{doi}" if doi else "https://example.org/paper/7"
    return AcquisitionRequest(
        slug=slug_for(doi, title),
        title=title,
        doi=doi,
        landing_url=landing_url,
        reason=AcquisitionReason.PAYWALLED,
        requested_at=datetime.now(UTC),
    )


class TestDoiPrefix:
    def test_the_prefix_is_the_registrant_part_before_the_first_slash(self) -> None:
        assert doi_prefix("10.1016/j.combustflame.2015.11.011") == "10.1016"

    def test_whitespace_and_case_do_not_change_the_prefix(self) -> None:
        assert doi_prefix("  10.1115/1.4007737 ") == "10.1115"

    def test_no_doi_yields_no_prefix(self) -> None:
        assert doi_prefix(None) is None

    def test_a_string_without_a_slash_is_not_a_doi_and_yields_no_prefix(self) -> None:
        assert doi_prefix("10.1016") is None


class TestPublisherLookup:
    def test_a_10_1016_doi_yields_the_elsevier_label(self) -> None:
        recipe = build_recipe(_request(), today=FRESH_DAY)
        assert recipe.publisher is not None
        assert "Elsevier" in recipe.publisher

    def test_every_registry_prefix_resolves_to_a_named_publisher(self) -> None:
        expected = {
            "10.1016",
            "10.1115",
            "10.1007",
            "10.1002",
            "10.1021",
            "10.1039",
            "10.1088",
            "10.3390",
            "10.1080",
            "10.1063",
            "10.2514",
        }
        assert expected <= set(PUBLISHER_RECIPES)
        for prefix in expected:
            recipe = build_recipe(_request(doi=f"{prefix}/test.2020.01.001"), today=FRESH_DAY)
            assert recipe.publisher, f"prefix {prefix} produced no publisher label"

    def test_an_unknown_doi_prefix_falls_through_to_generic_without_raising(self) -> None:
        request = _request(doi="10.99999/made.up.2020.001")
        recipe = build_recipe(request, today=FRESH_DAY)
        assert recipe.publisher is None
        assert recipe.formats == GENERIC_FORMATS
        assert recipe.open_url == request.landing_url
        assert recipe.save_as == f"inbox/{request.slug}.pdf"

    def test_a_request_with_no_doi_still_produces_an_actionable_recipe(self) -> None:
        """When automated retrieval failed without a DOI, ``landing_url`` is the
        paper's own URL (literature.py never files a request with neither), so the
        recipe must point the operator there and still name an exact drop filename."""
        request = _request(doi=None, landing_url="https://example.org/conf/paper-7")
        recipe = build_recipe(request, today=FRESH_DAY)
        assert recipe.publisher is None
        assert recipe.open_url == "https://example.org/conf/paper-7"
        assert recipe.save_as == f"inbox/{request.slug}.pdf"
        assert f"{request.slug}.si." in recipe.supplementary

    def test_no_recipe_claims_a_verified_click_path_in_v1(self) -> None:
        """DOI prefix -> publisher NAME is stable registrant data; a click-path is a
        UI claim nobody has verified. v1 ships the structure with every entry
        unverified, and the flag must survive into the built recipe."""
        for prefix, entry in PUBLISHER_RECIPES.items():
            assert entry.verified is False, f"registry entry {prefix} claims verification"
            assert entry.click_path is None, f"registry entry {prefix} ships a click-path"
        recipe = build_recipe(_request(), today=FRESH_DAY)
        assert recipe.verified is False


class TestElsevierFormats:
    def test_elsevier_instructs_pdf_and_never_sends_the_operator_after_xml(self) -> None:
        """ScienceDirect serves a TDM-reservation opt-out Carmel respects, and the
        sanctioned XML route needs an entitlement Carmel does not have -- so the
        Elsevier recipe must not send the operator hunting for XML."""
        recipe = build_recipe(_request(), today=FRESH_DAY)
        assert recipe.formats[0].label == "PDF"
        assert all(f.extension != "xml" for f in recipe.formats)

    def test_the_generic_xml_option_is_conditional_and_claims_only_preservation(self) -> None:
        """Guard against overclaiming: "solves header-to-column mapping" is a property
        Carmel does not have (no JATS table parser exists today). The honest claim is
        conditional availability plus structure preservation, nothing stronger."""
        (xml,) = [f for f in GENERIC_FORMATS if f.extension == "xml"]
        assert "explicitly offers" in xml.rationale
        assert "preserves table structure" in xml.rationale
        assert "100%" not in xml.rationale


class TestStaleness:
    def test_a_recipe_past_its_shelf_life_degrades_to_generic_and_says_so(self) -> None:
        """Registry entries are advisory and expiring: past ``stale_after`` the
        guidance is an assertion nobody has re-checked, so the operator gets the
        generic instructions plus a plain statement of why."""
        entry = PUBLISHER_RECIPES["10.1016"]
        stale_day = entry.verified_on + entry.stale_after + timedelta(days=1)
        recipe = build_recipe(_request(), today=stale_day)
        assert recipe.formats == GENERIC_FORMATS, (
            "a publisher recipe past its stale_after must degrade to the generic format instructions"
        )
        assert any("last verified" in note and "generic" in note for note in recipe.notes), (
            "a degraded recipe must say in its notes that it went stale and generic instructions apply"
        )

    def test_the_degradation_note_names_the_publisher_so_the_operator_knows_what_expired(self) -> None:
        entry = PUBLISHER_RECIPES["10.1016"]
        stale_day = entry.verified_on + entry.stale_after + timedelta(days=1)
        recipe = build_recipe(_request(), today=stale_day)
        assert any("Elsevier" in note for note in recipe.notes)

    def test_a_recipe_on_the_last_day_of_its_shelf_life_is_still_fresh(self) -> None:
        entry = PUBLISHER_RECIPES["10.1016"]
        last_fresh_day = entry.verified_on + entry.stale_after
        recipe = build_recipe(_request(), today=last_fresh_day)
        assert recipe.formats == entry.formats, (
            "a publisher recipe within its stale_after window must keep its own format instructions"
        )
        assert not any("last verified" in note for note in recipe.notes)
