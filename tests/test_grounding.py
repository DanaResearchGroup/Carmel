"""Tests for the deterministic grounding gate in carmel.services.grounding.

Fixtures are hand-built ExtractedText/TextSection instances (no fixture files),
constructed directly so each test controls exactly which raw-text region is
labeled "references" vs "body" without depending on extract.py's heading-detection
regex (which is a separate, already-tested concern).
"""

from __future__ import annotations

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
    MIN_NORMALIZED_QUOTE_LENGTH,
    _find_all_normalized,
    _fuzzy_search,
    _has_numeric_discrepancy,
    _present_outside_references,
    _section_for,
    _surname,
    check_evidence_spans,
    check_identity,
    find_quote,
    ground_finding,
    required_spans_for,
)


def _extracted(text: str, sections: list[TextSection] | None = None) -> ExtractedText:
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
    extracted = _extracted(text)
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


def test_has_numeric_discrepancy_false_when_diff_has_no_digits() -> None:
    assert _has_numeric_discrepancy("substantial result", "significant result") is False


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
    extracted = _extracted(text)
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
