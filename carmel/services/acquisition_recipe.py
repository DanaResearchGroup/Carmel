# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Per-publisher download recipes for the manual acquisition README.

The manual queue is the only working route for closed-access papers, and the README
this module feeds is the operator's ONLY instruction sheet. Two design decisions here
are load-bearing and were settled empirically, not aesthetically:

1. **The registry is keyed on DOI PREFIX, never on URL host.** A probe of a real
   operator queue found every request's ``landing_url`` was the DOI resolver
   (``https://doi.org/<doi>``) -- :mod:`carmel.services.literature` falls back to it
   whenever no better URL is known -- so a host-keyed registry would resolve to
   ``doi.org`` for every request and identify no publisher. The DOI prefix is the
   Crossref registrant identifier and discriminates correctly (``10.1016`` Elsevier,
   ``10.1115`` ASME, ...).

2. **No recipe ships an invented click-path.** A stale or guessed "click Download ->
   PDF" is worse than no click-path at all: it can walk the operator into the
   browser's "Save Page As", which the ingestion layer refuses outright (HTML and
   plain text are both rejected as primary documents). DOI prefix -> publisher NAME
   is stable registrant data and safe to ship; a click-path is a UI claim that must
   be VERIFIED against the live site before it may be recorded. v1 therefore ships
   every entry with ``verified=False`` and ``click_path=None``, and the README says
   plainly that the click-path has not been verified.

Recipes are additionally advisory and EXPIRING: an entry whose ``verified_on`` is
older than its ``stale_after`` degrades to the generic instructions and says so,
because guidance nobody has re-checked is an assertion, not knowledge.

Everything in this module is a pure function over the request: no network calls, no
LLM calls, no file I/O, and no hidden clock -- :func:`build_recipe` takes ``today``
explicitly so staleness is deterministic under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from carmel.schemas.acquisition import AcquisitionRequest


@dataclass(frozen=True)
class FormatPreference:
    """One acceptable download format, with the reason it is (or is only
    conditionally) worth getting. Order in a recipe's tuple IS the preference order."""

    label: str
    """Operator-facing name, e.g. ``"PDF"``."""
    extension: str
    """Filename extension (no dot) the drop should carry, e.g. ``"pdf"``."""
    rationale: str
    """One line telling the operator why / when to pick this format. Must never claim
    more than Carmel can honour -- e.g. XML "preserves table structure that the PDF
    loses" is a property of the markup, and is as strong as the claim may get."""


@dataclass(frozen=True)
class PublisherRecipe:
    """Advisory, expiring download guidance for one DOI prefix (Crossref registrant).

    ``verified`` / ``click_path`` describe the UI walk, which nobody has verified in
    v1; ``verified_on`` / ``stale_after`` describe the entry's shelf life -- past it,
    :func:`build_recipe` degrades to the generic instructions rather than keep
    asserting guidance that may have rotted under a publisher redesign.
    """

    publisher: str
    """Publisher name as the operator will see it, e.g. ``"Elsevier (ScienceDirect)"``."""
    formats: tuple[FormatPreference, ...]
    """Ordered format preferences for this publisher."""
    verified_on: date
    """When a human last checked this entry (in v1: the prefix->name mapping and the
    format guidance; never a click-path)."""
    sample_doi: str
    """A DOI carrying this prefix, kept so a future re-verifier has a concrete article
    to test the guidance against. The prefix is the load-bearing part; where no DOI
    from a live Carmel run was available, the sample is a pattern-typical DOI for the
    publisher's journals."""
    expected_output: str
    """What the operator should end up holding when the recipe worked."""
    stale_after: timedelta
    """Shelf life relative to ``verified_on``. Strictly older than this -> degrade."""
    notes: str = ""
    """Free-form guidance rendered with the request (e.g. Elsevier's XML caveat)."""
    verified: bool = False
    """Whether ``click_path`` was verified against the live site. Always False in v1;
    flips to True only when a human records a checked walk in ``click_path``."""
    click_path: str | None = None
    """The verified UI walk ("click X, then Y"), or None. Never populated unverified:
    an invented click-path is the one failure mode this registry must not have."""


@dataclass(frozen=True)
class AcquisitionRecipe:
    """The rendered-ready instruction set for one request: exactly where to go, what
    format to take, what to name it, and what to do about supplementary material."""

    publisher: str | None
    """Publisher name, or None when the DOI prefix is unknown / there is no DOI."""
    verified: bool
    """Whether a verified click-path backs this recipe (never, in v1)."""
    open_url: str
    """The exact URL the operator should open (the request's ``landing_url``: the DOI
    resolver when a DOI is known, otherwise the best address Carmel has)."""
    formats: tuple[FormatPreference, ...]
    """Ordered format preferences, each with its one-line rationale."""
    save_as: str
    """Exact drop filename for the preferred format, e.g. ``inbox/<slug>.pdf``."""
    supplementary: str
    """Full conditional instruction for supplementary material, naming the exact
    ``inbox/<slug>.si.<ext>`` convention :func:`carmel.services.acquisition.collect_inbox`
    binds."""
    notes: tuple[str, ...] = field(default=())
    """Free-form notes; the staleness-degradation message lives here when it applies."""


#: The generic, always-applicable format guidance: what to tell an operator when no
#: (fresh) publisher entry exists. PDF first -- it is the one format effectively every
#: publisher offers and Carmel ingests directly. Full-text XML is CONDITIONAL: offered
#: through the UI by some publishers only, and the claim stops at markup fidelity --
#: Carmel has no JATS table parser today, so nothing here may promise one.
GENERIC_FORMATS: tuple[FormatPreference, ...] = (
    FormatPreference(
        label="PDF",
        extension="pdf",
        rationale="the publisher's full-article PDF; every publisher offers it and Carmel ingests it directly",
    ),
    FormatPreference(
        label="Full-text XML",
        extension="xml",
        rationale=(
            "ONLY if the publisher explicitly offers a full-text XML download: save it unchanged -- "
            "it preserves table structure that the PDF loses"
        ),
    ),
)

#: The one shelf life used by every v1 entry. The content being aged here is the
#: prefix->name mapping (very stable Crossref registrant data) plus name-level format
#: guidance; a year keeps an annual re-verification honest without churning. Entries
#: that later gain a real click-path should carry a much shorter, per-entry value.
_V1_STALE_AFTER = timedelta(days=365)

#: The date the v1 entries were checked: prefix->publisher mappings confirmed against
#: Crossref registrant data, and the two prefixes observed in the live-syngas operator
#: queue (10.1016, 10.1115) confirmed against its real requests.
_V1_VERIFIED_ON = date(2026, 8, 1)


def _entry(
    publisher: str,
    sample_doi: str,
    *,
    formats: tuple[FormatPreference, ...] = GENERIC_FORMATS,
    notes: str = "",
) -> PublisherRecipe:
    return PublisherRecipe(
        publisher=publisher,
        formats=formats,
        verified_on=_V1_VERIFIED_ON,
        sample_doi=sample_doi,
        expected_output="the publisher's full-article file, byte-identical to what the site served",
        stale_after=_V1_STALE_AFTER,
        notes=notes,
    )


#: DOI prefix -> download recipe. Names only, deliberately: a prefix whose publisher
#: mapping is not certain stays OUT of this table -- an unknown prefix falls through
#: cleanly to the generic recipe, while a wrong name here would misdirect the operator.
PUBLISHER_RECIPES: dict[str, PublisherRecipe] = {
    "10.1016": _entry(
        "Elsevier (ScienceDirect)",
        # Real DOI from the live-syngas operator queue.
        "10.1016/j.ijhydene.2012.10.075",
        # PDF ONLY -- no XML entry. ScienceDirect marks its pages with a
        # text-and-data-mining reservation that Carmel respects (no scraping, no UA
        # spoofing), and Elsevier's sanctioned XML route requires an institutional TDM
        # entitlement Carmel does not have. Listing XML here would send the operator
        # hunting for a download that cannot be obtained through the site.
        formats=(
            FormatPreference(
                label="PDF",
                extension="pdf",
                rationale=(
                    "ScienceDirect's article XML is not downloadable through the site "
                    "(it needs a separate institutional text-and-data-mining entitlement), "
                    "so the full-article PDF is the file to get"
                ),
            ),
        ),
        notes=(
            "On ScienceDirect, download the full-article PDF; do not look for an XML "
            "download -- the site does not offer one."
        ),
    ),
    "10.1115": _entry(
        "ASME (American Society of Mechanical Engineers)",
        # Real DOI from the live-syngas operator queue.
        "10.1115/1.4007737",
    ),
    "10.1007": _entry("Springer (SpringerLink)", "10.1007/s00193-017-0784-y"),
    "10.1002": _entry("Wiley", "10.1002/kin.20603"),
    "10.1021": _entry("American Chemical Society (ACS)", "10.1021/acs.energyfuels.6b01204"),
    "10.1039": _entry("Royal Society of Chemistry (RSC)", "10.1039/c9cp03305j"),
    "10.1088": _entry("IOP Publishing", "10.1088/1361-6595/aa8688"),
    "10.3390": _entry("MDPI", "10.3390/en13123141"),
    "10.1080": _entry("Taylor & Francis", "10.1080/00102202.2016.1193499"),
    "10.1063": _entry("AIP Publishing", "10.1063/1.4798459"),
    "10.2514": _entry("AIAA (American Institute of Aeronautics and Astronautics)", "10.2514/1.B34934"),
}


def doi_prefix(doi: str | None) -> str | None:
    """Return the Crossref registrant prefix of ``doi`` (``10.1016`` of
    ``10.1016/j.fuel...``), or None when there is no DOI or the string carries no
    ``/`` and so cannot be a full DOI."""
    if not doi:
        return None
    cleaned = doi.strip().lower()
    prefix, sep, suffix = cleaned.partition("/")
    if not sep or not prefix or not suffix:
        return None
    return prefix


#: The SI naming convention, stated ONCE for the whole queue rather than repeated per
#: paper. Rendered by :func:`carmel.services.acquisition._readme_text` in the shared
#: "How to download" header; :func:`supplementary_instruction` then renders only the
#: per-paper fact that actually differs (the concrete stem), pointing back here.
#:
#: Deliberately split across separate backtick spans (``inbox/<slug>.si`` /
#: ``.si.1.<ext>`` / ...) rather than one ``inbox/<slug>.si.<ext>`` span: the latter
#: shape is what ``tests/test_acquisition.py``'s agreement test greps for, and it must
#: match only the per-paper line below (which carries a real slug), never this
#: generic, unsubstituted placeholder.
SUPPLEMENTARY_CONVENTION = (
    "Supplementary files: if the article page lists Supplementary Material (also "
    "called Supporting Information or an online Appendix), download each listed file "
    "UNCHANGED -- no re-saving, no converting -- and save it under `inbox/<slug>.si` "
    "plus the file's own extension: `.si.<ext>` for one file, or `.si.1.<ext>`, "
    "`.si.2.<ext>`, ... for several, where `<slug>` is the slug shown for that paper "
    "below. If the page lists none, nothing further is needed."
)


def supplementary_instruction(slug: str) -> str:
    """The terse, per-paper supplementary-material fact: only the concrete stem.

    Conditional by design: Crossref ``.s001``-style component records are dropped
    during search dedup, so Carmel holding no SI marker is NOT evidence the paper has
    no SI -- only the article page can say. The filename convention itself is stated
    once, in :data:`SUPPLEMENTARY_CONVENTION`; this renders only what differs per
    paper. The filename this names must match what
    :func:`carmel.services.acquisition.collect_inbox` binds (``<slug>.si.<ext>`` /
    ``<slug>.si.<n>.<ext>``); ``tests/test_acquisition.py`` drives a README-derived
    name through the real collector to hold the two in agreement.
    """
    return (
        f"Supplementary material (if the article page lists any): "
        f"`inbox/{slug}.si.<ext>` (see naming convention above)."
    )


def build_recipe(request: AcquisitionRequest, *, today: date) -> AcquisitionRecipe:
    """Build the download recipe for one queued request. Pure and deterministic.

    Args:
        request: The queued acquisition request.
        today: The date staleness is judged against. Explicit on purpose -- there is
            no hidden ``date.today()`` here, so expiry is unit-testable.

    Returns:
        The recipe: publisher-specific when the DOI prefix has a FRESH registry entry,
        otherwise the generic instructions (unknown prefix, no DOI, or a stale entry
        -- the last of which says so in ``notes``).
    """
    prefix = doi_prefix(request.doi)
    entry = PUBLISHER_RECIPES.get(prefix) if prefix is not None else None

    publisher: str | None = None
    verified = False
    formats = GENERIC_FORMATS
    notes: list[str] = []

    if entry is None:
        if prefix is not None:
            notes.append(f"DOI prefix `{prefix}` is not in Carmel's publisher registry; the generic steps above apply.")
        elif not request.doi:
            notes.append("No DOI is known for this paper; the link above is the best address Carmel has for it.")
    elif today - entry.verified_on > entry.stale_after:
        # Degrade to generic, and say so. The publisher NAME could survive (registrant
        # data is stable), but the guidance is what expired, and a half-degraded recipe
        # that keeps the label invites reading the stale guidance back into it -- so
        # the whole recipe reverts and the note carries the name instead.
        notes.append(
            f"A {entry.publisher} recipe exists for DOI prefix `{prefix}` but was last verified "
            f"{entry.verified_on:%Y-%m-%d}, past its {entry.stale_after.days}-day shelf life; "
            f"using the generic download instructions until it is re-verified."
        )
    else:
        publisher = entry.publisher
        verified = entry.verified
        formats = entry.formats
        if entry.notes:
            notes.append(entry.notes)

    return AcquisitionRecipe(
        publisher=publisher,
        verified=verified,
        open_url=request.landing_url,
        formats=formats,
        save_as=f"inbox/{request.slug}.{formats[0].extension}",
        supplementary=supplementary_instruction(request.slug),
        notes=tuple(notes),
    )
