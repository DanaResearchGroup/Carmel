"""Tests for the literature-research schemas."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from carmel.agents.budget import BudgetDimension, BudgetUsage
from carmel.agents.literature_agent import ProposedFinding, VerifierAssessment
from carmel.schemas.campaign import ReactorType
from carmel.schemas.literature import (
    ROOT_EXTRACTION_ID,
    STOP_REASON_FOR_DIMENSION,
    Citation,
    CredenceVerdict,
    EvidenceRef,
    ExperimentalBenchmarkPayload,
    FindingCategory,
    GroundingStatus,
    GroundingVerdict,
    LiteratureFinding,
    LiteratureReport,
    ObservableKind,
    PassRecord,
    PriorModelPayload,
    QMCalculationPayload,
    QMProperty,
    Quantity,
    QueryRecord,
    RejectedFinding,
    SpeciesRef,
    StopReason,
    StoredArtifact,
)
from carmel.services.artifacts import read_json, write_json


def _grounding() -> GroundingVerdict:
    return GroundingVerdict(
        status=GroundingStatus.GROUNDED_EXACT,
        grounded=True,
        match_ratio=1.0,
        identity_ok=True,
    )


def _citation(**kwargs: object) -> Citation:
    defaults: dict[str, object] = {"title": "A study of ignition delay"}
    defaults.update(kwargs)
    return Citation(**defaults)  # type: ignore[arg-type]


class TestCitation:
    def test_requires_one_identifier(self) -> None:
        with pytest.raises(ValidationError):
            Citation(title="x")

    def test_url_only_is_valid(self) -> None:
        c = Citation(title="x", url="https://example.com/paper")
        assert c.url == "https://example.com/paper"

    def test_source_id_only_is_valid(self) -> None:
        c = Citation(title="x", source_id="abc123")
        assert c.source_id == "abc123"

    def test_doi_normalizes_https_prefix(self) -> None:
        c = Citation(title="x", doi="https://doi.org/10.1000/ABC.123")
        assert c.doi == "10.1000/abc.123"

    def test_doi_normalizes_doi_colon_prefix(self) -> None:
        c = Citation(title="x", doi="DOI:10.1000/AbC.123")
        assert c.doi == "10.1000/abc.123"

    def test_doi_normalizes_mixed_case_no_prefix(self) -> None:
        c = Citation(title="x", doi="10.1000/AbC.123")
        assert c.doi == "10.1000/abc.123"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation(title="x", url="https://example.com", surprise="y")  # type: ignore[call-arg]


class TestFindingPayloadDiscriminatedUnion:
    def test_experimental_benchmark_selected(self) -> None:
        finding = _make_finding(_experimental_payload())
        assert isinstance(finding.payload, ExperimentalBenchmarkPayload)
        assert finding.payload.category == FindingCategory.EXPERIMENTAL_BENCHMARK

    def test_prior_model_selected(self) -> None:
        finding = _make_finding(_prior_model_payload())
        assert isinstance(finding.payload, PriorModelPayload)
        assert finding.payload.category == FindingCategory.PRIOR_MODEL

    def test_qm_calculation_selected(self) -> None:
        finding = _make_finding(_qm_payload())
        assert isinstance(finding.payload, QMCalculationPayload)
        assert finding.payload.category == FindingCategory.QM_CALCULATION

    def test_round_trip_via_dict_selects_correct_class(self) -> None:
        finding = _make_finding(_qm_payload())
        dumped = finding.model_dump(mode="json")
        restored = LiteratureFinding.model_validate(dumped)
        assert isinstance(restored.payload, QMCalculationPayload)

    def test_unknown_category_rejected(self) -> None:
        payload = _qm_payload().model_dump(mode="json")
        payload["category"] = "not_a_real_category"
        with pytest.raises(ValidationError):
            LiteratureFinding(
                finding_id="f1",
                payload=payload,  # type: ignore[arg-type]
                citation=_citation(doi="10.1/x"),
                verbatim_quote="quote",
                evidence=EvidenceRef(artifact_sha256="a" * 64, extraction_id=ROOT_EXTRACTION_ID),
                grounding=_grounding(),
            )


def _experimental_payload() -> ExperimentalBenchmarkPayload:
    return ExperimentalBenchmarkPayload(
        reactor_type=ReactorType.SHOCK_TUBE,
        observable=ObservableKind.IGNITION_DELAY_TIME,
        observable_raw="ignition delay time",
        temperature_range_K=(1000.0, 1500.0),
        measured=[Quantity(value=1.2, unit="ms", raw_text="1.2 ms")],
        species=[SpeciesRef(raw_name="OH", canonicalized=False)],
    )


def _prior_model_payload() -> PriorModelPayload:
    return PriorModelPayload(
        model_name="AramcoMech 3.0",
        n_species=500,
        n_reactions=3000,
    )


def _qm_payload() -> QMCalculationPayload:
    return QMCalculationPayload(
        level_of_theory="CCSD(T)/CBS",
        property=QMProperty.BARRIER_HEIGHT,
        value=Quantity(value=25.0, unit="kcal/mol"),
    )


def _make_finding(payload: object) -> LiteratureFinding:
    return LiteratureFinding(
        finding_id="f1",
        payload=payload,  # type: ignore[arg-type]
        citation=_citation(doi="10.1000/abc"),
        verbatim_quote="the ignition delay was 1.2 ms",
        evidence=EvidenceRef(artifact_sha256="a" * 64, extraction_id=ROOT_EXTRACTION_ID),
        grounding=_grounding(),
    )


class TestProposedFindingQuoteFloor:
    def test_short_verbatim_quote_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProposedFinding(
                payload=_qm_payload(),
                citation=_citation(doi="10.1000/abc"),
                verbatim_quote="too short",  # 9 chars, well under the 40-char floor
                source_url="https://example.com/paper.pdf",
            )

    def test_quote_at_floor_is_accepted(self) -> None:
        quote = "x" * 40
        finding = ProposedFinding(
            payload=_qm_payload(),
            citation=_citation(doi="10.1000/abc"),
            verbatim_quote=quote,
            source_url="https://example.com/paper.pdf",
        )
        assert finding.verbatim_quote == quote


class TestVerifierAssessmentMirrorsCredenceVerdict:
    def test_field_sets_match(self) -> None:
        # VerifierAssessment inherits from CredenceVerdict rather than
        # hand-transcribing its fields, so this should always hold. This test
        # guards against a future edit re-introducing a hand-copied, divergent
        # field set on either side.
        assert set(VerifierAssessment.model_fields) == set(CredenceVerdict.model_fields)

    def test_is_a_credence_verdict_subclass(self) -> None:
        assert issubclass(VerifierAssessment, CredenceVerdict)


class TestExtraForbid:
    def test_species_ref_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            SpeciesRef(raw_name="OH", bogus_field=1)  # type: ignore[call-arg]

    def test_quantity_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            Quantity(value=1.0, unit="K", bogus_field=1)  # type: ignore[call-arg]


class TestQuantityFiniteGuard:
    """A non-finite value can never be a trustworthy measured quantity, and a stored
    inf/nan would later become an ungroundable str(float) anchor ('inf'/'nan') in the
    grounding gate -- so it is rejected at construction."""

    def test_rejects_an_infinite_value(self) -> None:
        with pytest.raises(ValidationError):
            Quantity(value=float("inf"), unit="K")

    def test_rejects_a_negative_infinite_value(self) -> None:
        with pytest.raises(ValidationError):
            Quantity(value=float("-inf"), unit="K")

    def test_rejects_a_nan_value(self) -> None:
        with pytest.raises(ValidationError):
            Quantity(value=float("nan"), unit="K")

    def test_accepts_an_ordinary_finite_value(self) -> None:
        quantity = Quantity(value=850.0, unit="microseconds")
        assert quantity.value == 850.0

    def test_rejects_a_non_finite_uncertainty(self) -> None:
        with pytest.raises(ValidationError):
            Quantity(value=850.0, unit="microseconds", uncertainty=float("inf"))

    def test_accepts_an_ordinary_finite_uncertainty(self) -> None:
        quantity = Quantity(value=850.0, unit="microseconds", uncertainty=10.0)
        assert quantity.uncertainty == 10.0


class TestExperimentalBenchmarkPayloadFiniteGuard:
    """The same non-finite-value posture as :class:`TestQuantityFiniteGuard`, applied
    to every range/scalar condition field: a stored inf/nan here would likewise
    become an ungroundable str(float) anchor in the grounding gate."""

    def test_rejects_a_non_finite_temperature_range_bound(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentalBenchmarkPayload(
                reactor_type=ReactorType.SHOCK_TUBE,
                observable=ObservableKind.IGNITION_DELAY_TIME,
                observable_raw="ignition delay time",
                temperature_range_K=(1000.0, float("inf")),
            )

    def test_rejects_a_non_finite_pressure_range_bound(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentalBenchmarkPayload(
                reactor_type=ReactorType.SHOCK_TUBE,
                observable=ObservableKind.IGNITION_DELAY_TIME,
                observable_raw="ignition delay time",
                pressure_range_bar=(float("nan"), 2.0),
            )

    def test_rejects_a_non_finite_equivalence_ratio_range_bound(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentalBenchmarkPayload(
                reactor_type=ReactorType.SHOCK_TUBE,
                observable=ObservableKind.IGNITION_DELAY_TIME,
                observable_raw="ignition delay time",
                equivalence_ratio_range=(0.5, float("-inf")),
            )

    def test_rejects_a_non_finite_residence_time(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentalBenchmarkPayload(
                reactor_type=ReactorType.SHOCK_TUBE,
                observable=ObservableKind.IGNITION_DELAY_TIME,
                observable_raw="ignition delay time",
                residence_time_s=float("inf"),
            )

    def test_accepts_ordinary_finite_ranges_and_residence_time(self) -> None:
        payload = ExperimentalBenchmarkPayload(
            reactor_type=ReactorType.SHOCK_TUBE,
            observable=ObservableKind.IGNITION_DELAY_TIME,
            observable_raw="ignition delay time",
            temperature_range_K=(1000.0, 1500.0),
            pressure_range_bar=(1.0, 2.0),
            equivalence_ratio_range=(0.5, 1.5),
            residence_time_s=2.0,
        )
        assert payload.temperature_range_K == (1000.0, 1500.0)
        assert payload.residence_time_s == 2.0


class TestStopReasonForDimension:
    def test_covers_every_budget_dimension(self) -> None:
        for dimension in BudgetDimension:
            assert dimension in STOP_REASON_FOR_DIMENSION
            assert isinstance(STOP_REASON_FOR_DIMENSION[dimension], StopReason)

    def test_no_extra_dimensions(self) -> None:
        assert set(STOP_REASON_FOR_DIMENSION.keys()) == set(BudgetDimension)


class TestRejectedFinding:
    def test_valid(self) -> None:
        r = RejectedFinding(
            finding_id="f2",
            category=FindingCategory.QM_CALCULATION,
            citation_title="some paper",
            grounding=GroundingVerdict(
                status=GroundingStatus.QUOTE_NOT_FOUND,
                grounded=False,
                match_ratio=0.0,
            ),
            reason="quote could not be located in the stored artifact",
        )
        assert r.grounding.grounded is False


class TestLiteratureReportRoundTrip:
    def test_round_trips_through_write_json_read_json(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        finding = LiteratureFinding(
            finding_id="f1",
            payload=_experimental_payload(),
            citation=_citation(doi="https://doi.org/10.1000/XYZ"),
            verbatim_quote="the ignition delay was 1.2 ms at 1200 K",
            evidence=EvidenceRef(
                artifact_sha256="b" * 64,
                extraction_id=ROOT_EXTRACTION_ID,
                quote_start=10,
                quote_end=40,
                page=3,
            ),
            grounding=_grounding(),
            credence=CredenceVerdict(
                credence=0.8,
                provenance_score=0.9,
                quality_score=0.8,
                consistency_score=0.7,
                rationale="well-grounded, single non-canonical species",
                flags=["non_canonical_species"],
            ),
        )
        rejected = RejectedFinding(
            finding_id="f2",
            category=FindingCategory.PRIOR_MODEL,
            citation_title="unrelated paper",
            grounding=GroundingVerdict(
                status=GroundingStatus.REFERENCES_ONLY,
                grounded=False,
                match_ratio=0.5,
            ),
            reason="only found in references section",
        )
        artifact = StoredArtifact(
            sha256="b" * 64,
            source_url="https://example.com/paper.pdf",
            final_url="https://example.com/paper.pdf",
            content_type="application/pdf",
            n_bytes=12345,
            stored_at=now,
            extractor="pdf:pypdf",
            lossy=True,
        )
        report = LiteratureReport(
            report_id="r1",
            campaign_id="c1",
            created_at=now,
            passes=[
                PassRecord(
                    run_id="run1",
                    action_id="a1",
                    created_at=now,
                    model_name="mock",
                    stop_reason=StopReason.SELF_TERMINATED,
                    usage=BudgetUsage(
                        model_calls=3,
                        tokens=1200,
                        cost_usd=0.05,
                        fetches=2,
                        fetch_bytes=50000,
                        elapsed_s=12.5,
                    ),
                    warnings=["one species could not be canonicalized"],
                )
            ],
            queries=[QueryRecord(text="ethanol ignition delay shock tube", run_id="run1", action_id="a1")],
            artifacts=[artifact],
            findings=[finding],
            rejected=[rejected],
        )

        path = tmp_path / "literature_report.json"
        write_json(path, report)
        restored_data = read_json(path)
        restored = LiteratureReport.model_validate(restored_data)

        assert restored == report
