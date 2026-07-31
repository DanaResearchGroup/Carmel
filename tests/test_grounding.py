"""Tests for the deterministic grounding gate in carmel.services.grounding.

Fixtures are hand-built ExtractedText/TextSection instances (no fixture files),
constructed directly so each test controls exactly which raw-text region is
labeled "references" vs "body" without depending on extract.py's heading-detection
regex (which is a separate, already-tested concern).
"""

from __future__ import annotations

from difflib import SequenceMatcher

import pytest
from pydantic import ValidationError

from carmel.agents.tools.extract import (
    ExtractedText,
    TextSection,
    normalize_for_match,
    normalize_with_map,
    raw_span,
)
from carmel.schemas.campaign import ReactorType
from carmel.schemas.literature import (
    Citation,
    ExperimentalBenchmarkPayload,
    GroundingStatus,
    ObservableKind,
    PriorModelPayload,
    QMCalculationPayload,
    QMProperty,
    Quantity,
    SpeciesRef,
)
from carmel.services.grounding import (
    _QUOTE_MISS_EXPLANATIONS,
    MIN_NORMALIZED_QUOTE_LENGTH,
    QuoteMissReason,
    UnsupportedFindingPayloadError,
    _bounded_window,
    _find_all_normalized,
    _fuzzy_search,
    _has_semantic_discrepancy,
    _present_outside_references,
    _section_for,
    _surname,
    check_evidence_spans,
    check_identity,
    find_quote,
    find_quote_with_reason,
    ground_finding,
    required_spans_for,
    unreadable_reason,
)

#: The title carried by the citation almost every fixture below cites.
DEFAULT_FIXTURE_TITLE = "Ignition delay times study"


def _extracted(
    text: str,
    sections: list[TextSection] | None = None,
    *,
    front_matter_title: str | None = DEFAULT_FIXTURE_TITLE,
) -> ExtractedText:
    """Build an ``ExtractedText`` for a synthetic paper.

    A real paper states its own title on the front page, and :func:`check_identity`
    requires that -- the title is the corroboration that this document IS the cited
    work, with no surname-based escape from it. These fixtures exist to exercise
    *quote grounding*, not identity, so the title is supplied by default and only the
    identity tests vary it (``front_matter_title=None`` to omit it).

    Only prepended when the caller did not supply explicit ``sections``: those
    callers compute section offsets against the text they passed, and shifting it
    underneath them would silently mislabel every span.
    """
    if front_matter_title and sections is None and front_matter_title not in text:
        text = f"{front_matter_title}\n\n{text}"
    if sections is None:
        sections = [TextSection(label="body", start=0, end=len(text))]
    return ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=sections,
        extractor="text",
        lossy=False,
    )


def test_ground_finding_grounded_exact_experimental_benchmark() -> None:
    text = (
        "Abstract\n\n"
        "This is the abstract of the paper by Smith and Jones (2019). "
        "DOI: 10.1000/xyz123\n\n"
        "Introduction\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds "
        "in these shock tube experiments using O2 under stoichiometric conditions.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_EXACT
    assert verdict.grounded is True
    assert verdict.identity_ok is True
    assert verdict.missing_spans == []


def test_ground_finding_fabricated_quote_is_rejected() -> None:
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    quote = "This exact sentence was never written anywhere in this document at all"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND
    assert verdict.grounded is False


def test_ground_finding_quote_only_in_references_section() -> None:
    body = "Body content discussing other matters entirely, unrelated to the quoted material below.\n\n"
    references_heading = "References\n"
    reference_entry = (
        "Smith, J. (2019). The measured ignition delay time at 1200 K was 850 microseconds. Journal of Combustion."
    )
    text = body + references_heading + reference_entry
    references_start = len(body)
    sections = [
        TextSection(label="body", start=0, end=references_start),
        TextSection(label="references", start=references_start, end=len(text)),
    ]
    extracted = _extracted(text, sections)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.REFERENCES_ONLY
    assert verdict.grounded is False


def test_find_quote_matches_reflowed_text_exactly() -> None:
    # Extra whitespace runs, a hyphen-split word across a line break, and a ligature.
    raw_text = "The measured igni-\ntion   delay time at 1200 K was 850 microseconds, which is signiﬁcant.\n"
    extracted = _extracted(raw_text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds, which is significant."

    match = find_quote(extracted, quote)

    assert match is not None
    assert match.exact is True
    assert match.ratio == 1.0
    assert raw_text[match.start : match.end].replace("\n", "").strip() != ""


def test_find_quote_rejects_number_change_even_though_similar() -> None:
    raw_text = "The reaction was studied and the measured ignition delay time at 1200 K was 850 microseconds.\n"
    extracted = _extracted(raw_text)
    fabricated_quote = "The reaction was studied and the measured ignition delay time at 1500 K was 850 microseconds."

    match = find_quote(extracted, fabricated_quote)

    assert match is None


def test_ground_finding_number_change_rejected_end_to_end() -> None:
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The reaction was studied and the measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    fabricated_quote = "The reaction was studied and the measured ignition delay time at 1500 K was 850 microseconds."
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=fabricated_quote, extracted=extracted)

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND
    assert verdict.grounded is False


def test_check_identity_doi_mismatch_even_with_matching_title() -> None:
    text = (
        "Abstract\n\nThis paper by Smith, J. (2019), titled 'Ignition delay times study', "
        "presents new shock tube results. DOI: 10.9999/other-paper\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")

    assert check_identity(extracted, citation) is False


def test_ground_finding_identity_mismatch_when_doi_absent() -> None:
    text = (
        "Abstract\n\nThis paper by Smith, J. (2019), titled 'Ignition delay times study', "
        "presents new shock tube results. DOI: 10.9999/other-paper\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.IDENTITY_MISMATCH
    assert verdict.grounded is False
    assert verdict.identity_ok is False


def test_check_identity_doi_and_bare_surname_no_year_no_title_is_not_confirmed() -> None:
    """Spar round 5, P0: a review article that merely mentions the cited work's DOI
    in its body text, and which happens to contain the first author's bare surname
    somewhere (e.g. discussing a *different* paper by an author with the same common
    surname), must NOT be accepted as the cited paper. The review's own title never
    matches, and no corroborating year is present either -- only the DOI and a bare
    surname. Under the old ``doi_ok and (title_ok or author_ok)`` rule this passed,
    which is exactly the misattribution this regression pins closed."""
    text = (
        "Abstract\n\nThis review of combustion chemistry surveys many works. "
        "It cites DOI: 10.1000/xyz123 among numerous other references. "
        "Elsewhere, Smith and coworkers have separately studied unrelated polymer "
        "kinetics in a completely different context.\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text, front_matter_title=None)
    # The review's own title ("A Survey of Combustion Chemistry") never appears, and
    # the citation year (2019) is absent from the text entirely.
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")

    assert check_identity(extracted, citation) is False


def test_check_identity_doi_and_title_confirms_with_no_author_info() -> None:
    """Positive case: ``doi_ok and title_ok`` alone (no author corroboration at all)
    must still be sufficient -- the primary rule must not regress."""
    text = (
        "Abstract\n\nThis paper, titled 'Ignition delay times study', presents new "
        "shock tube results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(title="Ignition delay times study", authors=[], doi="10.1000/xyz123")

    assert check_identity(extracted, citation) is True


def test_check_identity_doi_surname_and_year_without_any_title_is_not_confirmed() -> None:
    """Spar round 5 P0, carried through round 6, now closed.

    A review article honestly carries every weak signal the old fallback accepted: it
    cites the primary work's DOI, it names the first author while discussing that
    work, and -- being a review -- it contains many four-digit years, of which the
    citation's is one. Under the old ``doi_ok and (title_ok or (author_ok and
    doi_year_ok))`` rule this returned True, so a quote taken from the REVIEW's own
    prose would have been recorded as fully grounded under the PRIMARY paper's
    citation.

    The cited title appears nowhere in this document, which is the whole point: the
    document is not that paper.
    """
    text = (
        "Abstract\n\nSmith and Jones (2019) present new shock tube combustion "
        "results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text, front_matter_title=None)
    citation = Citation(
        title="A title that never appears anywhere in this text",
        authors=["Smith, J."],
        year=2019,
        doi="10.1000/xyz123",
    )

    assert check_identity(extracted, citation) is False


def test_check_identity_confirms_a_title_damaged_by_extraction() -> None:
    """The case the removed surname+year fallback was actually reaching for.

    A title line that extracts imperfectly -- here a ligature lost and a hyphen
    broken across a line -- is still unmistakably the same title, so identity is
    confirmed by the fuzzy title path without any surname or year involvement.
    """
    text = (
        "Journal of Combustion\n\n"
        "Igni\ntion delay times of methane oxidation in a shock tue\n\n"
        "DOI: 10.1000/xyz123\n\nAbstract\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(
        title="Ignition delay times of methane oxidation in a shock tube",
        authors=[],
        doi="10.1000/xyz123",
    )

    assert check_identity(extracted, citation) is True


def test_check_identity_rejects_a_different_title_sharing_stock_vocabulary() -> None:
    """The fuzzy title path must not become a back door. Two different combustion
    papers share a great deal of boilerplate ("ignition delay times of ... in a shock
    tube"), so a near-miss on the distinguishing words is a DIFFERENT paper, not a
    damaged rendering of this one."""
    text = (
        "Journal of Combustion\n\n"
        "Ignition delay times of propane oxidation in a rapid compression machine\n\n"
        "DOI: 10.1000/xyz123\n\nAbstract\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(
        title="Ignition delay times of methane oxidation in a shock tube",
        authors=[],
        doi="10.1000/xyz123",
    )

    assert check_identity(extracted, citation) is False


def test_check_identity_rejects_a_title_that_differs_only_in_the_fuel_studied() -> None:
    """Spar round 7, P0. The previous test's two titles are far enough apart that the
    0.85 character ratio rejects them on its own. These two are NOT: they differ by two
    characters in one word, and the same group publishes both.

    The first assertion is the point of the test. It pins the measurement that showed
    the character ratio cannot carry this check -- the discriminating word is a few
    characters inside a long, otherwise identical string -- so that anyone who later
    deletes the token guard sees immediately that the ratio does NOT catch this.
    """
    cited = "Ignition delay times of methane oxidation in a shock tube"
    held = "Ignition delay times of methanol oxidation in a shock tube"
    ratio = SequenceMatcher(None, normalize_for_match(held), normalize_for_match(cited)).ratio()
    assert ratio > 0.95, "the fuzzy ratio alone admits this pair; the token guard is what refuses it"

    text = (
        "Journal of Combustion\n\n"
        f"{held}\n\n"
        "DOI: 10.1000/xyz123\n\nAbstract\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    citation = Citation(title=cited, authors=[], doi="10.1000/xyz123")

    assert check_identity(_extracted(text), citation) is False


def test_check_identity_rejects_an_erratum_that_reprints_the_original_title_and_doi() -> None:
    """Spar round 7, P0. An erratum satisfies BOTH conjuncts of the DOI rule honestly:
    it prints the original's DOI, and it reprints the original's full title by
    construction. Neither identity route can separate it from the paper it concerns.

    The reachable case is the corpus pass, not a mis-drop: an erratum legitimately held
    on its own merits, whose prose the agent then cites as the original paper. Without
    the marker gate a quote from the erratum's own text grounds under the original's
    citation.
    """
    text = (
        "Erratum to: Ignition delay times of methane oxidation in a shock tube\n"
        "[Combust. Flame 190 (2018) 100-110]\n\n"
        "DOI: 10.1000/xyz123\n\nCorrection\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds, not 950.\n"
    )
    citation = Citation(
        title="Ignition delay times of methane oxidation in a shock tube",
        authors=[],
        doi="10.1000/xyz123",
    )

    assert check_identity(_extracted(text, front_matter_title=None), citation) is False


def test_check_identity_confirms_a_paper_whose_own_title_announces_a_comment() -> None:
    """The marker gate must not swallow a paper genuinely titled "Comment on ...".

    Such papers are real, citable documents. The marker is suppressed when the CITED
    title contains it, so the gate distinguishes "this document is a comment on the
    paper you asked for" from "the paper you asked for is itself a comment".
    """
    title = "Comment on 'Laminar flame speeds of syngas mixtures'"
    text = (
        f"{title}\n\n"
        "DOI: 10.1000/xyz123\n\nAbstract\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    citation = Citation(title=title, authors=[], doi="10.1000/xyz123")

    assert check_identity(_extracted(text, front_matter_title=None), citation) is True


def test_check_identity_ignores_a_title_whose_window_spans_into_the_references() -> None:
    """Spar round 7. The companion to the test below, for the case that defeats it.

    A fuzzy window is exactly as long as the title and the scan is strided, so when a
    reference entry LEADS with the title, a window can begin in the body and run into
    the bibliography. Classifying that window by its start alone calls it "body" and
    confirms a title that occurs nowhere but the reference list -- the precise
    confusion the section check exists to catch. Both ends must be checked.
    """
    body = (
        "A Survey of Combustion Chemistry\n\n"
        "This review cites DOI: 10.1000/xyz123 among many other works.\n\n"
        "References\n"
    )
    references = "Ignition delay times of methane oxidation in a shock tube. Smith, J. (2019).\n"
    text = body + references
    sections = [
        TextSection(label="body", start=0, end=len(body)),
        TextSection(label="references", start=len(body), end=len(text)),
    ]
    citation = Citation(
        title="Ignition delay times of methane oxidation in a shock tube",
        authors=["Smith, J."],
        year=2019,
        doi="10.1000/xyz123",
    )

    assert check_identity(_extracted(text, sections), citation) is False


def test_check_identity_ignores_a_title_that_matches_only_in_the_references() -> None:
    """A review's reference list contains the primary paper's title EXACTLY, which is
    the highest-scoring window in the document. Identity must still be refused, and
    the fuzzy scan must not settle for that best window and stop."""
    body = (
        "A Survey of Combustion Chemistry\n\n"
        "This review cites DOI: 10.1000/xyz123 among many other works.\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n\n"
    )
    references = "References\nSmith, J. (2019). Ignition delay times of methane oxidation in a shock tube.\n"
    text = body + references
    sections = [
        TextSection(label="body", start=0, end=len(body)),
        TextSection(label="references", start=len(body), end=len(text)),
    ]
    extracted = _extracted(text, sections)
    citation = Citation(
        title="Ignition delay times of methane oxidation in a shock tube",
        authors=["Smith, J."],
        year=2019,
        doi="10.1000/xyz123",
    )

    assert check_identity(extracted, citation) is False


def test_ground_finding_identity_mismatch_when_citation_only_in_references() -> None:
    body = (
        "Body text describing an unrelated ignition delay time measurement. "
        "The measured ignition delay time at 1200 K was 850 microseconds under stoichiometric conditions.\n\n"
    )
    references_heading = "References\n"
    reference_entry = "Smith, J. (2019). Ignition delay times study. Journal of Combustion."
    text = body + references_heading + reference_entry
    references_start = len(body)
    sections = [
        TextSection(label="body", start=0, end=references_start),
        TextSection(label="references", start=references_start, end=len(text)),
    ]
    extracted = _extracted(text, sections)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    # No DOI on this citation, so identity falls back to title + author + year.
    citation = Citation(
        title="Ignition delay times study",
        authors=["Smith, J."],
        year=2019,
        url="https://example.org/paper",
    )
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.IDENTITY_MISMATCH
    assert verdict.grounded is False


def test_required_spans_for_empty_measured_is_the_empty_requirements_trap() -> None:
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[],
    )

    assert required_spans_for(payload) == []


def test_ground_finding_empty_measured_yields_spans_missing() -> None:
    text = (
        "Abstract\n\nSmith and Jones (2019) discuss ignition delay time. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time was reported qualitatively without a specific value.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time was reported qualitatively without a specific value"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False


def test_ground_finding_value_present_but_far_from_quote_is_spans_missing() -> None:
    filler = "x" * 1000
    text = (
        "Abstract\n\nSmith and Jones (2019) discuss QM results. DOI: 10.1000/xyz123\n\n"
        "The barrier height was computed using CCSD(T)/cc-pVTZ level of theory for reaction R1.\n"
        + filler
        + "\nElsewhere, far from the quote, the value 42.5 kcal/mol appears in an unrelated table.\n"
    )
    extracted = _extracted(text, front_matter_title="QM barrier heights")
    quote = "The barrier height was computed using CCSD(T)/cc-pVTZ level of theory for reaction R1"
    citation = Citation(title="QM barrier heights", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = QMCalculationPayload(
        level_of_theory="CCSD(T)/cc-pVTZ",
        property=QMProperty.BARRIER_HEIGHT,
        value=Quantity(value=42.5, unit="kcal/mol"),
        reaction_label="R1",
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert verdict.missing_spans != []


def test_bounded_window_forward_scan_falls_back_to_flat_window_when_no_boundary_is_found() -> None:
    """A sentence/row/paragraph boundary is not always nearby (e.g. a long unbroken
    run, or a table cell with no punctuation), and `_bounded_window` falls back to a
    flat `fallback`-character window on whichever side no boundary turns up within
    `2 * fallback` characters. Only the boundary-FOUND path is exercised elsewhere
    (via ordinary prose that reaches a period); this proves the forward/right-side
    scan-exhausted fallback specifically, by putting no boundary character anywhere
    within range on that side while the left side does find one, so the two sides can
    be told apart."""
    prefix = "Sentence one ends here. "
    quote = "QUOTE"
    suffix = "b" * 100  # no '.', '!', '?', or '\n' anywhere within 2 * fallback
    text = prefix + quote + suffix
    start = len(prefix)
    end = start + len(quote)

    window_start, window_end = _bounded_window(text, start, end, fallback=20)

    # Left side: the period right before `start` is found, so this is NOT the fallback.
    assert window_start == prefix.index(".") + 1
    # Right side: no boundary within 2*fallback=40 chars, so this is the flat fallback.
    assert window_end == end + 20


def test_check_evidence_spans_numeric_value_normalization() -> None:
    text = "The yield was measured to be 1 percent under these conditions, consistent with theory.\n"
    extracted = _extracted(text)
    match = find_quote(extracted, "The yield was measured to be 1 percent under these conditions")
    assert match is not None

    missing = check_evidence_spans(extracted, match, ["1.0", "percent"])

    assert missing == []


def test_check_evidence_spans_numeric_value_normalization_reverse() -> None:
    text = "The yield was measured to be 1.0 percent under these conditions, consistent with theory.\n"
    extracted = _extracted(text)
    match = find_quote(extracted, "The yield was measured to be 1.0 percent under these conditions")
    assert match is not None

    missing = check_evidence_spans(extracted, match, ["1", "percent"])

    assert missing == []


def test_ground_finding_no_artifact() -> None:
    citation = Citation(title="Some paper", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = PriorModelPayload(model_name="GRI-Mech 3.0", n_species=53)

    verdict = ground_finding(payload=payload, citation=citation, quote="anything", extracted=None)

    assert verdict.status == GroundingStatus.NO_ARTIFACT
    assert verdict.grounded is False


def _pdf_extracted(text: str, *, pages: int) -> ExtractedText:
    """An ExtractedText that reports itself as coming from a paginated PDF."""
    return ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=[TextSection(label="body", start=0, end=len(text), page=1)],
        page_count=pages,
        extractor="pdf:pypdf",
        lossy=False,
    )


def test_scanned_pdf_with_no_text_layer_is_unreadable_not_fabrication() -> None:
    """A scanned/image-only PDF extracts to ~nothing; blaming the agent for that
    records a false fabrication accusation in the decision log."""
    extracted = _pdf_extracted("\n\n \n", pages=14)
    citation = Citation(title="Turbulent combustion study", authors=["Kesten, A."], year=1970, doi="10.1000/kes70")
    payload = PriorModelPayload(model_name="Some model", n_species=10)

    verdict = ground_finding(payload=payload, citation=citation, quote="a perfectly honest quote", extracted=extracted)

    assert verdict.status == GroundingStatus.ARTIFACT_UNREADABLE
    assert verdict.grounded is False
    assert any("scanned" in r or "no text" in r.lower() for r in verdict.reasons)


def test_space_lost_extraction_is_unreadable_not_fabrication() -> None:
    """Some PDFs encode fonts without space glyphs, so pypdf returns run-together
    text. An honestly-transcribed quote can never match it."""
    text = (
        "Mechanismandkineticsoftheisothermalthermodegradationof\n"
        "ethylene-propylene-diene(EPDM)elastomers\n"
        "Thethermaldegradationbehaviourofarangeofelastomerswasstudiedusing\n"
        "thermogravimetricanalysisoverthewholerangeofcompositionsreported\n"
    ) * 4
    extracted = _pdf_extracted(text, pages=7)
    citation = Citation(title="Mechanism and kinetics", authors=["Gamlin, C."], year=2001, doi="10.1000/gam01")
    payload = PriorModelPayload(model_name="Some model", n_species=10)

    verdict = ground_finding(
        payload=payload,
        citation=citation,
        quote="The thermal degradation behaviour of a range of elastomers was studied",
        extracted=extracted,
    )

    assert verdict.status == GroundingStatus.ARTIFACT_UNREADABLE
    assert verdict.grounded is False
    assert any("word spacing" in r for r in verdict.reasons)


def test_mostly_clean_document_with_one_mangled_block_is_not_unreadable() -> None:
    """Space loss is measured blockwise, but a single damaged block in an otherwise
    healthy paper must NOT excuse a missing quote -- otherwise one bad table becomes a
    blanket amnesty for fabrication."""
    clean = "The measured ignition delay time was reported for each condition tested. " * 200
    mangled = "Thermogravimetricanalysisoverthewholerangeofcompositionsreported" * 30
    extracted = _pdf_extracted(clean + mangled, pages=8)

    verdict = ground_finding(
        payload=PriorModelPayload(model_name="Some model", n_species=10),
        citation=Citation(title="Paper", authors=["Smith, J."], year=2019, doi="10.1000/ok"),
        quote="a quote that is simply not present in this document anywhere",
        extracted=extracted,
    )

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND


def test_readable_pdf_still_reports_fabrication_not_unreadable() -> None:
    """The unreadable diagnosis must not become a blanket excuse: a healthy document
    with a fabricated quote is still QUOTE_NOT_FOUND."""
    text = (
        "The measured ignition delay time at 1200 K was 850 microseconds in these "
        "shock tube experiments under stoichiometric conditions. " * 12
    )
    extracted = _pdf_extracted(text, pages=2)
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = PriorModelPayload(model_name="Some model", n_species=10)

    verdict = ground_finding(
        payload=payload,
        citation=citation,
        quote="This exact sentence was never written anywhere in this document at all",
        extracted=extracted,
    )

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND
    assert verdict.grounded is False


def test_short_html_artifact_is_not_treated_as_unreadable() -> None:
    """The page-density check is PDF-only; a legitimately short HTML page must still
    catch a fabricated quote rather than being excused as unreadable."""
    extracted = _extracted("A short page. The rate constant was reported as 1.2e13 cm3/mol/s.")

    verdict = ground_finding(
        payload=PriorModelPayload(model_name="Some model", n_species=10),
        citation=Citation(title="Short note", authors=["Smith, J."], year=2019, doi="10.1000/short"),
        quote="a quote that simply is not present in this short page",
        extracted=extracted,
    )

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND


def test_ground_finding_warns_when_quote_near_tail_with_no_references_section() -> None:
    padding = "Unrelated filler content padding out the document body text. " * 40
    tail_quote = "The final measured ignition delay time at 1200 K was 850 microseconds"
    text = (
        "Abstract\n\nSmith and Jones (2019) discuss combustion. DOI: 10.1000/xyz123\n\n"
        + padding
        + tail_quote
        + " in a shock tube using O2 under stoichiometric conditions.\n"
    )
    # Only a single "body" section covering the whole document: no labelled
    # references section, even though this quote sits at the very end of the doc.
    extracted = _extracted(text)
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=tail_quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_EXACT
    assert verdict.grounded is True
    assert any("WARNING" in reason and "references" in reason.lower() for reason in verdict.reasons)


def test_find_quote_fuzzy_fallback_accepted_without_numeric_discrepancy() -> None:
    # A single non-numeric word swap: exact match fails, but the ratio clears the
    # default threshold and the diff never touches a digit, so the fuzzy fallback
    # is accepted with exact=False.
    text = "The measured ignition delay time was found to be quite substantial under these conditions.\n"
    extracted = _extracted(text)
    quote = "The measured ignition delay time was found to be quite significant under these conditions."

    match = find_quote(extracted, quote)

    assert match is not None
    assert match.exact is False
    assert 0.92 <= match.ratio < 1.0


def test_ground_finding_grounded_fuzzy_status() -> None:
    text = (
        "The measured ignition delay time was found to be quite substantial under these conditions, "
        "using the ignition delay time model with 53 species, per Smith and Jones (2019) combustion "
        "results. DOI: 10.1000/xyz123\n"
        # Title placed AFTER the quote rather than as front matter: this fixture turns
        # on the fuzzy quote scan, whose sliding window is strided from offset 0, so
        # shifting the quote's offset re-aligns the windows and changes what the scan
        # finds. Identity needs the title present somewhere outside references; it does
        # not care where.
        f"\n{DEFAULT_FIXTURE_TITLE}\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time was found to be quite significant under these conditions,"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    # PriorModelPayload's required anchors are the model name and n_species; both
    # are chosen to be present in the SAME sentence as the quote (the bounded
    # evidence window is now sentence-bounded), so this reaches GROUNDED_FUZZY
    # rather than SPANS_MISSING.
    payload = PriorModelPayload(model_name="ignition delay time", n_species=53)
    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_FUZZY
    assert verdict.grounded is True


def test_required_spans_for_prior_model_with_and_without_extras() -> None:
    payload_with_extra = PriorModelPayload(model_name="GRI-Mech 3.0", n_species=53)
    assert required_spans_for(payload_with_extra) == ["GRI-Mech 3.0", "53"]

    payload_without_extra = PriorModelPayload(model_name="GRI-Mech 3.0")
    assert required_spans_for(payload_without_extra) == []


def test_check_identity_no_authors_returns_false() -> None:
    text = "The paper titled Ignition delay times study was published in 2019.\n"
    extracted = _extracted(text)
    citation = Citation(title="Ignition delay times study", authors=[], year=2019, url="https://example.org/paper")

    assert check_identity(extracted, citation) is False


def test_surname_without_comma_uses_last_token() -> None:
    assert _surname("John Smith") == "Smith"
    assert _surname("Smith, J.") == "Smith"
    assert _surname("Cher") == "Cher"


def test_present_outside_references_empty_term_is_false() -> None:
    extracted = _extracted("Some body text with no matches of interest here.\n")
    assert _present_outside_references(extracted, "") is False


def test_find_all_normalized_empty_needle_returns_empty_list() -> None:
    assert _find_all_normalized("some haystack text", "") == []


def test_section_for_defaults_to_body_when_uncovered() -> None:
    # Sections list that does not cover the queried offset at all.
    sections = [TextSection(label="body", start=0, end=5)]
    label, page = _section_for(sections, 100)
    assert label == "body"
    assert page is None


def test_fuzzy_search_returns_none_for_empty_needle_or_haystack() -> None:
    assert _fuzzy_search("", "something", 0.92) is None
    assert _fuzzy_search("something", "", 0.92) is None


def test_has_semantic_discrepancy_false_when_diff_has_no_digits_or_negation() -> None:
    # "substantial"/"significant" is an ordinary PDF-damage-style near-miss with no
    # digit change, negation, or antonym involved -- must stay unflagged.
    assert _has_semantic_discrepancy("substantial result", "significant result") is False


def test_find_quote_returns_none_for_blank_quote() -> None:
    extracted = _extracted("Some real body text here.\n")
    assert find_quote(extracted, "    ") is None


def test_find_quote_raw_offsets_recover_expected_raw_substring() -> None:
    # Offsets are recovered via extract.normalize_with_map + extract.raw_span (the
    # same primitive that produced extracted.normalized in the first place), so
    # there is no separate reconstruction to fall out of sync and no proportional-
    # scaling fallback path anymore. This proves the raw offsets returned by
    # find_quote actually land on the raw text that, once re-normalized, matches
    # the quote — for text with reflowed whitespace, hyphenation, and a ligature.
    raw_text = "The measured igni-\ntion   delay time at 1200 K was 850 microseconds, which is signiﬁcant.\n"
    extracted = _extracted(raw_text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds, which is significant."

    match = find_quote(extracted, quote)

    assert match is not None
    assert match.exact
    raw_slice = extracted.text[match.start : match.end]
    assert normalize_for_match(raw_slice) == normalize_for_match(quote)

    # Cross-check against the underlying primitives directly.
    normalized, index_map = normalize_with_map(extracted.text)
    idx = normalized.find(normalize_for_match(quote))
    expected_start, expected_end = raw_span(index_map, idx, idx + len(normalize_for_match(quote)), len(extracted.text))
    assert (match.start, match.end) == (expected_start, expected_end)


# --- Regression tests for the Defect 1/2/3 fixes -----------------------------------


def test_reactor_type_and_conditions_mismatch_rejected() -> None:
    """Exact counter-example: artifact is a shock tube at 1200 K / 1 atm, but the
    finding claims a JSR at 1500 K / 20 bar for the same 850 microsecond value --
    this must NOT ground even though the measured value and unit both match."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In these shock tube experiments at 1200 K and 1 atm, the ignition delay time "
        "was 850 microseconds, using O2 as the oxidizer.\n"
    )
    extracted = _extracted(text)
    quote = "the ignition delay time was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.JSR,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        temperature_range_K=(1500.0, 1500.0),
        pressure_range_bar=(20.0, 20.0),
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert verdict.missing_spans != []


def test_correct_except_pressure_alone_is_rejected() -> None:
    """Every field is corroborated except the pressure bound, which is absent from
    the artifact text entirely -- this alone must sink the finding."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In these shock tube experiments at 1200 K, the ignition delay time was 850 "
        "microseconds, using O2 as the oxidizer.\n"
    )
    extracted = _extracted(text)
    quote = "the ignition delay time was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        temperature_range_K=(1200.0, 1200.0),
        pressure_range_bar=(20.0, 20.0),
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert any("20" in span or "20.0" in span for span in verdict.missing_spans)


def test_anchor_in_neighbouring_table_row_not_same_sentence_rejected() -> None:
    """An anchor that only appears in a neighbouring table row -- separated from
    the quote by a sentence/row boundary -- must not count as corroboration."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "Table 1: conditions were shock tube, 1200 K, 1 atm.\n"
        "The ignition delay time was 850 microseconds for this species.\n"
    )
    extracted = _extracted(text)
    quote = "The ignition delay time was 850 microseconds for this species"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    # "shock tube" and "O2" sit in the PREVIOUS sentence/row, not the quote's own
    # sentence, so the sentence-bounded window must not pick them up.
    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert verdict.missing_spans != []


def test_conditions_genuinely_stated_near_quote_still_grounds() -> None:
    """Guard against over-rejection: when the reactor type, conditions, and species
    ARE genuinely stated in the same sentence as the quote, the finding still
    grounds."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In these shock tube experiments at 1200 K and 1 atm, the ignition delay time "
        "was 850 microseconds using O2 as the oxidizer.\n"
    )
    extracted = _extracted(text)
    quote = "the ignition delay time was 850 microseconds using O2 as the oxidizer"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        temperature_range_K=(1200.0, 1200.0),
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_EXACT
    assert verdict.grounded is True
    assert verdict.missing_spans == []


def test_shock_tube_and_st_abbreviation_both_satisfy_reactor_anchor() -> None:
    """Both "shock-tube" and the abbreviation "ST" must satisfy the shock-tube
    reactor-type anchor."""
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")

    text_hyphen = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In this shock-tube study, the ignition delay time was 850 microseconds using O2.\n"
    )
    extracted_hyphen = _extracted(text_hyphen)
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )
    verdict_hyphen = ground_finding(
        payload=payload,
        citation=citation,
        quote="the ignition delay time was 850 microseconds using O2",
        extracted=extracted_hyphen,
    )
    assert verdict_hyphen.status == GroundingStatus.GROUNDED_EXACT

    text_abbrev = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In this ST facility, the ignition delay time was 850 microseconds using O2.\n"
    )
    extracted_abbrev = _extracted(text_abbrev)
    verdict_abbrev = ground_finding(
        payload=payload,
        citation=citation,
        quote="the ignition delay time was 850 microseconds using O2",
        extracted=extracted_abbrev,
    )
    assert verdict_abbrev.status == GroundingStatus.GROUNDED_EXACT


def test_review_article_dois_mention_without_title_or_author_fails_identity() -> None:
    """A review article that mentions another paper's DOI in body text -- without
    that paper's title or first author appearing anywhere -- must not establish
    identity (DOI is necessary but not sufficient)."""
    text = (
        "Abstract\n\nThis review surveys recent shock-tube ignition delay studies. "
        "One relevant dataset carries DOI: 10.1000/xyz123, reporting an ignition delay "
        "time of 850 microseconds at 1200 K.\n"
    )
    extracted = _extracted(text, front_matter_title=None)
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")

    assert check_identity(extracted, citation) is False

    quote = "reporting an ignition delay time of 850 microseconds at 1200 K"
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )
    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)
    assert verdict.status == GroundingStatus.IDENTITY_MISMATCH
    assert verdict.grounded is False


def test_unlabelled_bibliography_detected_structurally_rejects_quote() -> None:
    """A dense run of citation-pattern lines with NO references heading at all must
    still be recognized as bibliography-like and reject a quote landing inside it."""
    bib_lines = "\n".join(
        f"Author{i}, J. (20{10 + i}). Some paper title {i}. Journal of Combustion, "
        f"{i}:{i * 10}-{i * 10 + 5}. doi:10.1000/xyz{i}"
        for i in range(10)
    )
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "Introduction\n\nSome unrelated body prose discussing the setup in general terms.\n\n"
        f"{bib_lines}\n"
    )
    extracted = _extracted(text)
    # Quote is drawn verbatim from inside the dense, unlabelled bibliography run.
    quote = "Author3, J. (2013). Some paper title 3. Journal of Combustion, 3:30-35. doi:10.1000/xyz3"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = PriorModelPayload(model_name="GRI-Mech 3.0", n_species=53)

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.REFERENCES_ONLY
    assert verdict.grounded is False


def test_find_quote_rejects_degenerate_quote_that_normalizes_below_floor() -> None:
    # 61 raw characters (comfortably above literature_agent's raw 40-char floor) that
    # is almost entirely whitespace/punctuation and collapses, after normalization, to
    # 17 characters -- well under MIN_NORMALIZED_QUOTE_LENGTH. Without the floor, this
    # degenerate remainder would trivially "exact match" nearly any document.
    degenerate_quote = "     ...      ,,,        .        x        .        ,,,      "
    assert len(degenerate_quote) >= 40
    normalized = normalize_for_match(degenerate_quote)
    assert len(normalized) < MIN_NORMALIZED_QUOTE_LENGTH

    text = "Some unrelated document body that happens to contain an x somewhere in its prose.\n"
    extracted = _extracted(text)

    match = find_quote(extracted, degenerate_quote)

    assert match is None


def test_ground_finding_rejects_degenerate_quote_end_to_end() -> None:
    degenerate_quote = "     ...      ,,,        .        x        .        ,,,      "
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=degenerate_quote, extracted=extracted)

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND
    assert verdict.grounded is False


def test_check_identity_short_surname_does_not_spuriously_confirm_identity() -> None:
    # "Xu" normalizes to "xu" (2 chars), which coincidentally occurs as a substring of
    # "exuberant" -- not as a mention of the author. Title and year both genuinely
    # appear in the text. Before the MIN_IDENTITY_TERM_LENGTH floor, this coincidental
    # substring hit would have satisfied author_ok and (combined with real title/year
    # matches) spuriously confirmed identity for a citation whose author was never
    # actually referenced in the document.
    text = (
        "Abstract\n\nThis 2019 paper, titled 'Ignition delay times study', presents new shock "
        "tube results and reports an exuberant level of detail in the combustion measurements.\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    citation = Citation(
        title="Ignition delay times study",
        authors=["Xu, L."],
        year=2019,
        url="https://example.org/paper",
    )

    surname_norm = normalize_for_match(_surname(citation.authors[0]))
    assert len(surname_norm) < 4
    # Sanity-check the coincidental substring really is present in the raw text --
    # otherwise this test would pass for the wrong reason.
    assert surname_norm in extracted.normalized

    assert check_identity(extracted, citation) is False


# --- P1-2/P1-3/P1-4 critical reproductions and fixes -------------------------------


def test_find_quote_rejects_negation_deletion_even_though_similar() -> None:
    """P1-2: deleting "not" is a one-word edit with a very high character-similarity
    ratio, yet it inverts the sentence's meaning entirely -- the old numeric-only
    guard (_has_numeric_discrepancy) had no way to catch this."""
    raw_text = "The additive did not increase the ignition delay time under these conditions.\n"
    extracted = _extracted(raw_text)
    fabricated_quote = "The additive did increase the ignition delay time under these conditions."

    match = find_quote(extracted, fabricated_quote)

    assert match is None


def test_find_quote_rejects_antonym_substitution_even_though_similar() -> None:
    """P1-2: "increases" -> "decreases" is a high-ratio near-miss (they share a long
    common suffix) that flips the claimed direction of an effect."""
    raw_text = "The additive increases the ignition delay time under these standard conditions.\n"
    extracted = _extracted(raw_text)
    fabricated_quote = "The additive decreases the ignition delay time under these standard conditions."

    match = find_quote(extracted, fabricated_quote)

    assert match is None


def test_find_quote_rejects_negating_prefix_addition_even_though_similar() -> None:
    """P1-2: adding the "un-" prefix ("stable" -> "unstable") is a tiny character
    edit that inverts meaning."""
    raw_text = "The mixture was stable under these conditions during the entire experiment.\n"
    extracted = _extracted(raw_text)
    fabricated_quote = "The mixture was unstable under these conditions during the entire experiment."

    match = find_quote(extracted, fabricated_quote)

    assert match is None


def test_ground_finding_negation_deletion_rejected_end_to_end() -> None:
    """P1-2 end-to-end: before the fix this fabricated negation-dropped quote would
    be accepted via the fuzzy fallback and could proceed to a GROUNDED_* verdict."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The additive did not increase the ignition delay time under these conditions.\n"
    )
    extracted = _extracted(text)
    fabricated_quote = "The additive did increase the ignition delay time under these conditions."
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = PriorModelPayload(model_name="ignition delay time additive study", n_species=1)

    verdict = ground_finding(payload=payload, citation=citation, quote=fabricated_quote, extracted=extracted)

    assert verdict.status == GroundingStatus.QUOTE_NOT_FOUND
    assert verdict.grounded is False


def test_has_semantic_discrepancy_true_for_negation_token_deletion() -> None:
    assert (
        _has_semantic_discrepancy("the additive did increase the delay", "the additive did not increase the delay")
        is True
    )


def test_has_semantic_discrepancy_true_for_antonym_substitution() -> None:
    assert _has_semantic_discrepancy("the additive increases the delay", "the additive decreases the delay") is True


def test_has_semantic_discrepancy_true_for_negating_prefix_addition() -> None:
    assert _has_semantic_discrepancy("the mixture was stable", "the mixture was unstable") is True


def test_has_semantic_discrepancy_false_for_ordinary_words_sharing_a_negating_prefix() -> None:
    # "increase"/"increased" both start with "in", but neither is the bare stem of
    # the other with the prefix removed ("crease"/"creased" are not real words in
    # this text), so this must NOT be flagged as a negating-prefix edit.
    assert _has_semantic_discrepancy("measurements increase over time", "measurements increased over time") is False


def test_has_semantic_discrepancy_true_for_numeric_change_unchanged_behavior() -> None:
    # The original numeric-discrepancy behavior must be fully preserved.
    assert _has_semantic_discrepancy("1200 K", "1500 K") is True


def test_has_semantic_discrepancy_false_when_word_sets_are_identical_despite_a_character_diff() -> None:
    # A comma is a non-"equal" opcode with no digit in it, so the loop runs past the
    # digit check without returning; the two strings then reduce to the exact same
    # lowercased word set, so `diff_words` is empty and the function must fall
    # through to `return False` at that pass-through, not reach the negation/prefix/
    # antonym checks below it (which would also return False, but for the wrong
    # reason -- this proves the *empty diff_words* branch specifically).
    assert _has_semantic_discrepancy("the delay was measured, carefully.", "the delay was measured carefully.") is (
        False
    )


def test_ground_finding_degraded_reload_rejected_not_grounded() -> None:
    """P1-3: when extracted.json is missing, evidence.load_artifact_text's degraded
    reload sets sections=[] and lossy=True. Before the fix, ground_finding never
    consulted `lossy`, so this silently produced a GROUNDED_EXACT pass instead of a
    fail-closed rejection."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds in these shock "
        "tube experiments using O2 under stoichiometric conditions.\n"
    )
    extracted = ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=[],
        extractor="text",
        lossy=True,
    )
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.ARTIFACT_DEGRADED
    assert verdict.grounded is False


def test_ground_finding_lossy_but_sections_retained_is_not_degraded() -> None:
    """Guard against over-rejection: `lossy=True` is also set for ordinary,
    acceptable truncation where sections were retained (e.g. a long PDF cut off
    partway through). That must NOT be treated the same as a degraded reload."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds in these shock "
        "tube experiments using O2 under stoichiometric conditions.\n"
    )
    text = f"{DEFAULT_FIXTURE_TITLE}\n\n{text}"
    extracted = ExtractedText(
        text=text,
        normalized=normalize_for_match(text),
        sections=[TextSection(label="body", start=0, end=len(text))],
        extractor="text",
        lossy=True,
    )
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_EXACT
    assert verdict.grounded is True


def test_required_spans_for_only_anchors_first_measured_value_is_the_bug() -> None:
    """P1-4: a second, fabricated ``measured`` entry must also be required -- not
    silently ignored because only ``measured[0]`` was anchored."""
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds"), Quantity(value=9999.0, unit="parsecs")],
    )

    required = required_spans_for(payload)

    assert "9999.0" in required
    assert "parsecs" in required


def test_required_spans_for_requires_each_species_individually() -> None:
    """P1-4: ``species`` must not be a single any-one-suffices tuple anchor -- a
    single genuine species must not vouch for an arbitrary number of fabricated
    ones alongside it."""
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2"), SpeciesRef(raw_name="Unobtainium")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    required = required_spans_for(payload)

    assert "O2" in required
    assert "Unobtainium" in required
    assert not any(isinstance(r, tuple) and "O2" in r and "Unobtainium" in r for r in required)


def test_ground_finding_fabricated_extra_measurement_is_not_silently_corroborated() -> None:
    """P1-4 end-to-end: before the fix, only measured[0] (850 microseconds, genuinely
    present) was anchored, so a second fabricated measurement sailed through
    unchecked."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds in these shock "
        "tube experiments using O2 under stoichiometric conditions.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds"), Quantity(value=9999.0, unit="parsecs")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert any("9999.0" in span for span in verdict.missing_spans)


def test_ground_finding_fabricated_extra_species_is_not_silently_corroborated() -> None:
    """P1-4 end-to-end: before the fix, ``species`` was checked as an any-one-of
    tuple, so the genuinely-present "O2" alone let a fabricated "Unobtainium" ride
    along unchecked."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds in these shock "
        "tube experiments using O2 under stoichiometric conditions.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2"), SpeciesRef(raw_name="Unobtainium")],
        measured=[Quantity(value=850.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False
    assert any("Unobtainium" in span for span in verdict.missing_spans)


def test_ground_finding_multiple_genuine_measured_and_species_still_grounds() -> None:
    """Guard against over-rejection: when there really are two measured values and
    two species, both genuinely present near the quote, the finding must still
    ground -- the per-entry requirement must not become an unsatisfiable AND over
    entries that were never meant to share one sentence."""
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "In these shock tube experiments, the measured ignition delay times were 850 "
        "microseconds for O2 and 920 microseconds for CH4 under stoichiometric conditions.\n"
    )
    extracted = _extracted(text)
    quote = (
        "the measured ignition delay times were 850 microseconds for O2 and 920 "
        "microseconds for CH4 under stoichiometric conditions"
    )
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2"), SpeciesRef(raw_name="CH4")],
        measured=[Quantity(value=850.0, unit="microseconds"), Quantity(value=920.0, unit="microseconds")],
    )

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.GROUNDED_EXACT
    assert verdict.grounded is True
    assert verdict.missing_spans == []


def test_experimental_benchmark_payload_rejects_too_many_measured_entries() -> None:
    with pytest.raises(ValidationError):
        ExperimentalBenchmarkPayload(
            reactor_type=ReactorType.SHOCK_TUBE,
            observable=ObservableKind.IGNITION_DELAY_TIME,
            observable_raw="ignition delay time",
            measured=[Quantity(value=float(i), unit="microseconds") for i in range(9)],
        )


def test_experimental_benchmark_payload_rejects_too_many_species() -> None:
    with pytest.raises(ValidationError):
        ExperimentalBenchmarkPayload(
            reactor_type=ReactorType.SHOCK_TUBE,
            observable=ObservableKind.IGNITION_DELAY_TIME,
            observable_raw="ignition delay time",
            species=[SpeciesRef(raw_name=f"species{i}") for i in range(21)],
        )


# --- Informational fixes: exhaustiveness, coverage gaps -----------------------------


def test_required_spans_for_unsupported_payload_type_raises_named_error() -> None:
    with pytest.raises(UnsupportedFindingPayloadError):
        required_spans_for(object())  # type: ignore[arg-type]


def test_ground_finding_unsupported_payload_type_fails_closed_as_spans_missing() -> None:
    text = (
        "Abstract\n\nSmith and Jones (2019) report combustion results. DOI: 10.1000/xyz123\n\n"
        "The measured ignition delay time at 1200 K was 850 microseconds.\n"
    )
    extracted = _extracted(text)
    quote = "The measured ignition delay time at 1200 K was 850 microseconds"
    citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")

    verdict = ground_finding(payload=object(), citation=citation, quote=quote, extracted=extracted)  # type: ignore[arg-type]

    assert verdict.status == GroundingStatus.SPANS_MISSING
    assert verdict.grounded is False


def test_required_spans_for_prior_model_with_n_reactions_only() -> None:
    payload = PriorModelPayload(model_name="GRI-Mech 3.0", n_reactions=325)

    assert required_spans_for(payload) == ["GRI-Mech 3.0", "325"]


def test_required_spans_for_prior_model_with_mechanism_url_only() -> None:
    payload = PriorModelPayload(model_name="GRI-Mech 3.0", mechanism_url="https://example.org/mech.yaml")

    assert required_spans_for(payload) == ["GRI-Mech 3.0", "https://example.org/mech.yaml"]


def test_required_spans_for_qm_calculation_empty_floor_returns_empty_list() -> None:
    payload = QMCalculationPayload(
        level_of_theory="CCSD(T)/cc-pVTZ",
        property=QMProperty.BARRIER_HEIGHT,
        value=Quantity(value=42.5, unit="kcal/mol"),
    )

    assert required_spans_for(payload) == []


# --- Finding 2 (spar round 5, P1): required_spans_for must be exhaustive over ------
# --- every load-bearing populated payload field, per category. ---------------------


def test_required_spans_for_experimental_benchmark_anchors_residence_time_apparatus_and_n_data_points() -> None:
    """``residence_time_s``, ``apparatus``, and ``n_data_points`` are numeric/
    identifying claims that must be individually anchored when populated -- an
    earlier version silently omitted them, letting an LLM fabricate them freely."""
    payload = ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        species=[SpeciesRef(raw_name="O2")],
        measured=[Quantity(value=850.0, unit="microseconds")],
        residence_time_s=0.5,
        apparatus="high-pressure shock tube",
        n_data_points=12,
    )

    required = required_spans_for(payload)

    assert "0.5" in required
    assert "high-pressure shock tube" in required
    assert "12" in required


def test_required_spans_for_prior_model_anchors_fuel_species_and_validation_targets() -> None:
    """``fuel_species`` and ``validation_targets`` are identifying/factual claims
    (which species the model covers, which datasets it was validated against) and
    must be anchored individually when populated."""
    payload = PriorModelPayload(
        model_name="GRI-Mech 3.0",
        n_species=53,
        fuel_species=[SpeciesRef(raw_name="CH4"), SpeciesRef(raw_name="C2H6")],
        validation_targets=["Dooley et al. shock tube ignition delays"],
    )

    required = required_spans_for(payload)

    assert "CH4" in required
    assert "C2H6" in required
    assert "Dooley et al. shock tube ignition delays" in required


def test_required_spans_for_prior_model_conditions_note_is_deliberately_not_anchored() -> None:
    """``conditions_note`` is free-text narrative commentary, not a discrete
    verifiable claim like a species name or a count -- it is deliberately NOT
    required as a literal anchor (see report/commit message for the judgement)."""
    payload = PriorModelPayload(
        model_name="GRI-Mech 3.0",
        n_species=53,
        conditions_note="This is a narrative aside that need not appear verbatim near any quote.",
    )

    required = required_spans_for(payload)

    assert "This is a narrative aside that need not appear verbatim near any quote." not in required


def test_required_spans_for_qm_calculation_anchors_property_surface_form_and_software() -> None:
    """``property`` (an identifying fact about what was computed) and ``software``
    must be anchored when populated. ``property`` has no raw-text companion field
    (unlike ``observable``/``observable_raw``), so it is anchored via a surface-form
    synonym table, analogous to ``_REACTOR_TYPE_TERMS``."""
    payload = QMCalculationPayload(
        level_of_theory="CCSD(T)/cc-pVTZ",
        property=QMProperty.BOND_DISSOCIATION_ENERGY,
        value=Quantity(value=42.5, unit="kcal/mol"),
        reaction_label="R1",
        software="Gaussian 16",
    )

    required = required_spans_for(payload)

    assert any(isinstance(r, tuple) and "bond dissociation energy" in r for r in required), required
    assert "Gaussian 16" in required


def test_required_spans_for_qm_calculation_other_property_has_no_literal_anchor() -> None:
    """Judgement call: ``QMProperty.OTHER`` is a catch-all with no verbatim surface
    form to search for (unlike a specific property name), so it is deliberately
    exempted from the property anchor rather than requiring the literal word
    "other" to appear near the quote, which would spuriously reject genuine
    findings."""
    payload = QMCalculationPayload(
        level_of_theory="CCSD(T)/cc-pVTZ",
        property=QMProperty.OTHER,
        value=Quantity(value=42.5, unit="kcal/mol"),
        reaction_label="R1",
    )

    required = required_spans_for(payload)

    assert not any(isinstance(r, tuple) and "other" in r for r in required)
    assert "other" not in required


def test_required_spans_for_is_exhaustive_over_every_finding_category() -> None:
    """Structural regression: walk the ``FindingPayload`` discriminated union and
    confirm every declared category has representative coverage in
    ``required_spans_for`` for its claim-bearing fields. A new category added to the
    union without a corresponding branch in ``required_spans_for`` must fail this
    test (via the ``UnsupportedFindingPayloadError`` fail-closed path) rather than
    silently grounding with zero required anchors."""
    import typing

    from carmel.schemas.literature import FindingPayload

    union_type = typing.get_args(FindingPayload)[0]
    declared_categories = set(typing.get_args(union_type))

    representative_payloads: dict[type, FindingPayload] = {
        ExperimentalBenchmarkPayload: ExperimentalBenchmarkPayload(
            reactor_type=ReactorType.SHOCK_TUBE,
            observable=ObservableKind.IGNITION_DELAY_TIME,
            observable_raw="ignition delay time",
            species=[SpeciesRef(raw_name="O2")],
            measured=[Quantity(value=850.0, unit="microseconds")],
            temperature_range_K=(1000.0, 1200.0),
            pressure_range_bar=(1.0, 2.0),
            equivalence_ratio_range=(0.5, 1.5),
            residence_time_s=0.5,
            apparatus="high-pressure shock tube",
            n_data_points=12,
        ),
        PriorModelPayload: PriorModelPayload(
            model_name="GRI-Mech 3.0",
            n_species=53,
            n_reactions=325,
            fuel_species=[SpeciesRef(raw_name="CH4")],
            mechanism_url="https://example.org/mech.yaml",
            validation_targets=["Dooley et al. shock tube ignition delays"],
        ),
        QMCalculationPayload: QMCalculationPayload(
            level_of_theory="CCSD(T)/cc-pVTZ",
            property=QMProperty.BOND_DISSOCIATION_ENERGY,
            value=Quantity(value=42.5, unit="kcal/mol"),
            species=[SpeciesRef(raw_name="CH3")],
            reaction_label="R1",
            software="Gaussian 16",
        ),
    }

    # Every category declared in the union must have representative coverage above --
    # this is what makes the test fail loudly if a category is added without also
    # updating this structural test and (by extension) required_spans_for.
    assert declared_categories == set(representative_payloads)

    expected_claim_fragments: dict[type, list[str]] = {
        ExperimentalBenchmarkPayload: [
            "ignition delay time",
            "O2",
            "850.0",
            "microseconds",
            "1000.0",
            "1200.0",
            "1.0",
            "2.0",
            "0.5",
            "1.5",
            "high-pressure shock tube",
            "12",
        ],
        PriorModelPayload: [
            "GRI-Mech 3.0",
            "53",
            "325",
            "CH4",
            "https://example.org/mech.yaml",
            "Dooley et al. shock tube ignition delays",
        ],
        QMCalculationPayload: [
            "CCSD(T)/cc-pVTZ",
            "42.5",
            "kcal/mol",
            "Gaussian 16",
        ],
    }

    for payload_type, payload in representative_payloads.items():
        required = required_spans_for(payload)
        flattened: list[str] = []
        for r in required:
            if isinstance(r, tuple):
                flattened.extend(r)
            else:
                flattened.append(r)
        for fragment in expected_claim_fragments[payload_type]:
            assert fragment in flattened, (payload_type, fragment, required)


def test_unreadable_reason_pdf_unavailable() -> None:
    extracted = ExtractedText(
        text="",
        normalized="",
        sections=[],
        extractor="pdf:unavailable",
        lossy=True,
    )

    reason = unreadable_reason(extracted)

    assert reason is not None
    assert "pdf" in reason.lower()


def test_ground_finding_pdf_unavailable_is_artifact_unreadable_not_fabrication() -> None:
    extracted = ExtractedText(
        text="",
        normalized="",
        sections=[],
        extractor="pdf:unavailable",
        lossy=True,
    )
    quote = "Any claimed quote at all"
    citation = Citation(title="Some paper", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
    payload = PriorModelPayload(model_name="GRI-Mech 3.0", n_species=53)

    verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=extracted)

    assert verdict.status == GroundingStatus.ARTIFACT_UNREADABLE
    assert verdict.grounded is False


class TestQuoteMissReasonsAreDistinguishable:
    """All four misses reject, but the decision log must say WHICH.

    "the agent altered a measured value" and "the agent invented this wholesale" are
    different findings about a model's reliability, and a researcher reads this log to
    tell them apart. Before this, all four produced one generic fabrication accusation --
    which for the too-short case was simply untrue.
    """

    SOURCE = (
        "In these experiments the ignition delay time was not significantly affected by "
        "the additive at 1200 K and 10 bar, across the full range of equivalence ratios."
    )

    def test_too_short_is_not_reported_as_fabrication(self) -> None:
        match, reason = find_quote_with_reason(_extracted(self.SOURCE), "1200 K")
        assert match is None
        assert reason is QuoteMissReason.TOO_SHORT

    def test_absent_text_is_reported_as_fabrication(self) -> None:
        match, reason = find_quote_with_reason(
            _extracted(self.SOURCE),
            "the catalyst was regenerated by calcination at 800 K for six hours",
        )
        assert match is None
        assert reason is QuoteMissReason.NOT_FOUND

    def test_negation_deletion_is_reported_as_misrepresentation_not_invention(self) -> None:
        """The near-identical passage IS in the document -- the quote misstates it."""
        match, reason = find_quote_with_reason(
            _extracted(self.SOURCE),
            "the ignition delay time was significantly affected by the additive at 1200 K and 10 bar",
        )
        assert match is None
        assert reason is QuoteMissReason.SEMANTIC_DISCREPANCY

    def test_the_three_reasons_produce_three_different_operator_explanations(self) -> None:
        explanations = {
            _QUOTE_MISS_EXPLANATIONS[reason]
            for reason in (
                QuoteMissReason.TOO_SHORT,
                QuoteMissReason.NOT_FOUND,
                QuoteMissReason.SEMANTIC_DISCREPANCY,
            )
        }
        assert len(explanations) == 3
        assert "NOT evidence that the quote was fabricated" in _QUOTE_MISS_EXPLANATIONS[QuoteMissReason.TOO_SHORT]
