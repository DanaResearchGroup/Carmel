"""What a replay owes about values that no ref supports.

A `MeasuredValue` grounds its number and its unit and says nothing about its
quantity kind; an `Uncertainty` carries no `SourceRef` on any field at all.
These tests pin the narrow rule that reports exactly those gaps -- and, just as
importantly, pin the places it must stay SILENT, because a rule that fires
everywhere buries the three claims that decide anything under a hundred that do
not.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    Coordinate,
    DataPoint,
    DatasetEnvelope,
    MeasuredValue,
    Observation,
    Series,
    Uncertainty,
    UncertaintyBasis,
    UncertaintyKind,
    UncertaintyScale,
    iter_uncertainties,
)
from carmel.services.dataset_replay import (
    ReplayOutcome,
    SemanticGap,
    _dataset_uncertainty_sites,
    _kinds_admitting_unit,
    _quantity_kind_claim,
    _reconcile_uncertainty_sites,
    _uncertainty_claims,
    replay_condition_set,
)
from carmel.services.units import TABLE_V1, QuantityKind
from tests.test_condition_set_replay import (
    _embedded_table_v1,
    _minimal_condition_set,
    _pressure,
    _scalar_claim_with_uncertainty,
)

_ABSENT = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)


def _value(unit_raw: str, kind: QuantityKind, unit_normalized: str | None = None) -> MeasuredValue:
    base = _pressure("1.5", 0).model_dump()
    base.update(
        quantity_kind=kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized if unit_normalized is not None else unit_raw,
    )
    return MeasuredValue(**base)


class TestTheWildcardKindCannotBeCounted:
    """`QuantityKind.OTHER` accepts ANY unit string, so counting it as a
    candidate would score every value in the corpus as ambiguous and turn the
    narrow rule back into the blanket one it replaced."""

    def test_other_admits_a_unit_no_table_has_ever_heard_of(self) -> None:
        from carmel.services import units

        units.normalize_unit(QuantityKind.OTHER, "furlongs per fortnight", table=TABLE_V1)

    def test_other_is_absent_from_the_candidate_set_even_for_a_real_unit(self) -> None:
        admitting = _kinds_admitting_unit(TABLE_V1, "atm")
        assert QuantityKind.OTHER not in admitting
        assert admitting == (QuantityKind.PRESSURE,)

    def test_a_pinned_unit_would_look_ambiguous_if_other_were_counted(self) -> None:
        """The regression this exclusion prevents, stated as a fact about the
        table rather than about the code: `atm` is accepted by exactly one real
        kind AND by OTHER, so an implementation that counted OTHER would see
        two and report a claim for every value ever recorded."""
        from carmel.services import units

        units.normalize_unit(QuantityKind.OTHER, "atm", table=TABLE_V1)
        assert len(_kinds_admitting_unit(TABLE_V1, "atm")) == 1


class TestAUnitThatPinsItsQuantityIsNotReported:
    """The scope cut. Where the recorded table admits the unit for exactly one
    kind, `verify_measured_value_unit` has already closed the question and a
    claim would be noise."""

    @pytest.mark.parametrize("unit_raw,kind", [("atm", QuantityKind.PRESSURE), ("K", QuantityKind.TEMPERATURE)])
    def test_no_claim_for_an_unambiguous_unit(self, unit_raw: str, kind: QuantityKind) -> None:
        assert _quantity_kind_claim("p", _value(unit_raw, kind)) is None


class TestAUnitSharedBySeveralQuantitiesIsReported:
    def test_a_dimensionless_unit_yields_a_claim_naming_every_candidate(self) -> None:
        claim = _quantity_kind_claim("p", _value("-", QuantityKind.MOLE_FRACTION, unit_normalized="1"))
        assert claim is not None
        assert claim.gap is SemanticGap.NO_SUPPORT_OFFERED
        assert claim.claim_path == "p.quantity_kind"
        assert claim.claim == "mole_fraction"
        # Every rival kind is named, so a reader can see WHAT it might have been
        # instead of only that something was unchecked.
        for rival in ("mass_fraction", "equivalence_ratio", "relative_uncertainty"):
            assert rival in claim.reason

    def test_the_claim_offers_no_support_paths(self) -> None:
        """`NO_SUPPORT_OFFERED` owns no span. This is what keeps the new claims
        structurally unable to disturb the report's span arithmetic."""
        claim = _quantity_kind_claim("p", _value("percent", QuantityKind.MASS_FRACTION, unit_normalized="%"))
        assert claim is not None
        assert claim.support_paths == ()


class TestAnUndecidableTableFailsClosed:
    def test_an_unresolvable_table_sha_yields_a_claim_rather_than_silence(self) -> None:
        """If the table cannot be resolved, ambiguity cannot be decided -- and
        undecided must not read as unambiguous."""
        base = _pressure("1.5", 0).model_dump()
        base["conversion_table_sha256"] = "0" * 64
        value = MeasuredValue.model_construct(**base)
        claim = _quantity_kind_claim("p", value)
        assert claim is not None
        assert claim.gap is SemanticGap.NO_SUPPORT_OFFERED
        assert "no known" in claim.reason


class TestUncertaintyReportsAssertionsAndNotRefusals:
    """A recorded "not stated" is the honest answer the narrow slice exists to
    allow. Reporting it as an unsupported claim would invert that."""

    def test_a_concrete_kind_basis_and_scale_all_yield_claims(self) -> None:
        claims = _uncertainty_claims(
            "u",
            Uncertainty(
                kind=UncertaintyKind.CI_95,
                basis=UncertaintyBasis.ABSOLUTE,
                scale=UncertaintyScale.LINEAR,
                upper=_ABSENT,
                lower=_ABSENT,
            ),
        )
        assert {claim.claim_path for claim in claims} == {"u.kind", "u.basis", "u.scale"}
        assert {claim.gap for claim in claims} == {SemanticGap.NO_SUPPORT_OFFERED}

    @pytest.mark.parametrize(
        "kind", [UncertaintyKind.UNKNOWN, UncertaintyKind.UNSPECIFIED_PERCENTAGE]
    )
    def test_the_not_stated_sentinels_are_not_reported(self, kind: UncertaintyKind) -> None:
        claims = _uncertainty_claims(
            "u", Uncertainty(kind=kind, basis=_ABSENT, scale=_ABSENT, upper=_ABSENT, lower=_ABSENT)
        )
        assert claims == ()

    def test_an_absent_basis_and_scale_are_not_reported(self) -> None:
        claims = _uncertainty_claims(
            "u",
            Uncertainty(
                kind=UncertaintyKind.STD_DEV, basis=_ABSENT, scale=_ABSENT, upper=_ABSENT, lower=_ABSENT
            ),
        )
        assert {claim.claim_path for claim in claims} == {"u.kind"}


class TestTheHandWrittenInventoryIsReconciledAgainstTheWalk:
    """The inventory and the walk are written separately so that each can catch
    the other going stale. These pin that the check actually fires -- in BOTH
    directions."""

    def test_an_uncertainty_the_inventory_missed_is_a_finding(self) -> None:
        envelope = _minimal_condition_set(
            Path(tempfile.mkdtemp()),
            scalar_claims=(_scalar_claim_with_uncertainty(),),
            conversion_tables=(_embedded_table_v1(),),
        )
        findings = _reconcile_uncertainty_sites(envelope, named_paths=set())
        assert findings, "the walk reaches an uncertainty the empty inventory never named"
        assert all(f.category is ReplayOutcome.FAILED for f in findings)
        assert any("gone stale" in f.reason for f in findings)

    def test_an_inventory_entry_the_walk_cannot_reach_is_a_finding(self) -> None:
        envelope = _minimal_condition_set(Path(tempfile.mkdtemp()))
        findings = _reconcile_uncertainty_sites(envelope, named_paths={"scalar_claims[7].uncertainty"})
        assert [f.ref_path for f in findings] == ["scalar_claims[7].uncertainty"]
        assert "not in this envelope" in findings[0].reason

    def test_a_faithful_inventory_produces_no_finding(self) -> None:
        envelope = _minimal_condition_set(
            Path(tempfile.mkdtemp()),
            scalar_claims=(_scalar_claim_with_uncertainty(),),
            conversion_tables=(_embedded_table_v1(),),
        )
        walked = {path for path, _ in iter_uncertainties(envelope)}
        assert walked, "fixture must actually carry an uncertainty for this to mean anything"
        assert _reconcile_uncertainty_sites(envelope, named_paths=walked) == ()


class TestTheDatasetInventoryReachesEveryUncertaintyBearingSite:
    """The dataset fixtures in this repo happen to carry NO uncertainty at all,
    so nothing else in the suite exercises `_dataset_uncertainty_sites` -- a
    mutation blinding it to observations survived the whole suite. This pins
    all three of its branches against the generic walk directly.

    Built with `model_construct` deliberately: both the inventory and the walk
    are pure structural traversals, so validation is irrelevant to what is
    being compared, and a fully valid multi-point envelope would bury the
    comparison in fixture machinery.
    """

    @staticmethod
    def _envelope_with_uncertainties_everywhere() -> DatasetEnvelope:
        unc = Uncertainty(
            kind=UncertaintyKind.STD_DEV,
            basis=UncertaintyBasis.ABSOLUTE,
            scale=UncertaintyScale.LINEAR,
            upper=_ABSENT,
            lower=_ABSENT,
        )
        def build(model: type[BaseModel], **fields: object) -> BaseModel:
            """`model_construct` with every OTHER field explicitly ``None``.

            Both walks read fields by name, so a field left unset raises
            `AttributeError` instead of being skipped. Filling them from
            `model_fields` rather than by hand keeps this fixture from going
            stale the next time one of these models grows a field -- which is
            the very failure the inventory/walk reconciliation exists to catch,
            and it would be perverse for its own test to succumb to it.
            """
            defaults: dict[str, object] = {name: None for name in model.model_fields}
            return model.model_construct(**(defaults | fields))

        constant = build(Coordinate, value=_pressure("1.5", 0), uncertainty=unc)
        coordinate = build(Coordinate, value=_pressure("1.5", 0), uncertainty=unc)
        observation = build(Observation, value=_pressure("1.5", 0), uncertainty=unc)
        point = build(DataPoint, coordinates=(coordinate,), observations=(observation,))
        series = build(Series, constants=(constant,), points=(point,))
        return cast(DatasetEnvelope, build(DatasetEnvelope, series=(series,)))

    def test_the_inventory_finds_exactly_what_the_walk_finds(self) -> None:
        envelope = self._envelope_with_uncertainties_everywhere()
        inventory = {path for path, _ in _dataset_uncertainty_sites(envelope)}
        walked = {path for path, _ in iter_uncertainties(envelope)}
        assert inventory == walked
        assert _reconcile_uncertainty_sites(envelope, inventory) == ()

    @pytest.mark.parametrize(
        "expected",
        [
            "series[0].constants[0].uncertainty",
            "series[0].points[0].coordinates[0].uncertainty",
            "series[0].points[0].observations[0].uncertainty",
        ],
    )
    def test_each_bearing_site_is_reached_individually(self, expected: str) -> None:
        """Parametrised per branch so that blinding the inventory to ONE of
        constants, coordinates or observations fails on its own line rather
        than hiding inside a set comparison that another branch still satisfies.
        """
        envelope = self._envelope_with_uncertainties_everywhere()
        assert expected in {path for path, _ in _dataset_uncertainty_sites(envelope)}


class TestTheClaimsReachTheReportWithoutDisturbingItsArithmetic:
    def test_a_condition_set_carrying_an_uncertainty_reports_its_obligations(
        self, tmp_path: Path
    ) -> None:
        envelope = _minimal_condition_set(
            tmp_path,
            scalar_claims=(_scalar_claim_with_uncertainty(),),
            conversion_tables=(_embedded_table_v1(),),
        )
        report = replay_condition_set(tmp_path, envelope)
        paths = {claim.claim_path for claim in report.unchecked_semantic_claims}
        assert "scalar_claims[0].uncertainty.kind" in paths
        assert "scalar_claims[0].uncertainty.basis" in paths
        assert "scalar_claims[0].uncertainty.scale" in paths
        # The report still adds up: a NO_SUPPORT_OFFERED claim owns no span, so
        # it can never push checked + support_only past the total. Constructing
        # the report at all is the assertion -- __post_init__ raises otherwise.
        assert (
            report.checked_char_spans + report.support_only_char_spans <= report.total_char_spans
        )

    def test_the_overall_verdict_is_downgraded_but_the_evidence_verdict_is_not(
        self, tmp_path: Path
    ) -> None:
        """The whole point of the two scopes. The spans still re-slice clean;
        what is unverifiable is the MEANING no ref supports."""
        envelope = _minimal_condition_set(
            tmp_path,
            scalar_claims=(_scalar_claim_with_uncertainty(),),
            conversion_tables=(_embedded_table_v1(),),
        )
        report = replay_condition_set(tmp_path, envelope)
        assert report.unchecked_semantic_claims
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE


class TestTheDatasetPathCarriesTheSameObligations:
    """Emitting these on only one replay path would leave `overall_outcome`'s
    promise overclaimed on the other, with nothing to show a consumer why."""

    def test_replay_envelope_reports_a_ref_less_quantity_kind(self, tmp_path: Path) -> None:
        from carmel.services.dataset_replay import replay_envelope
        from tests.test_dataset_replay import _produce_and_load

        _stored, loaded = _produce_and_load(tmp_path)
        report = replay_envelope(tmp_path, loaded)
        assert [claim.claim_path for claim in report.unchecked_semantic_claims] == [
            "series[0].points[0].observations[0].value.quantity_kind"
        ]
        assert report.evidence_outcome is ReplayOutcome.VERIFIED
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE


class TestABrokenObjectGetsAVerdictNotATraceback:
    """The public replay entry points take an already-constructed object, so
    `model_construct` can hand them a field that is not the enum the schema
    types it as. Reaching for `.value` there used to raise out of the middle of
    a replay. A replayer owes a verdict about broken input."""

    def test_a_non_enum_quantity_kind_still_produces_a_claim(self) -> None:
        base = _pressure("1.5", 0).model_dump()
        base.update(quantity_kind=None, unit_raw="-", unit_normalized="1")
        claim = _quantity_kind_claim("p", MeasuredValue.model_construct(**base))
        assert claim is not None
        assert claim.claim == "None"

    def test_a_non_enum_uncertainty_field_still_produces_a_claim(self) -> None:
        unc = Uncertainty.model_construct(
            kind="std_dev", basis=None, scale=UncertaintyScale.LINEAR, upper=_ABSENT, lower=_ABSENT
        )
        claims = _uncertainty_claims("u", unc)
        assert {claim.claim_path for claim in claims} == {"u.kind", "u.basis", "u.scale"}
        assert all(claim.claim.strip() for claim in claims)


class TestACleanDatasetStillReportsVerifiedOverall:
    """The positive counterweight. Without it, a wiring bug that ALWAYS added a
    semantic claim would leave every private test green and every public report
    permanently unverifiable."""

    def test_single_kind_units_and_no_concrete_uncertainty_verify_overall(
        self, tmp_path: Path
    ) -> None:
        envelope = _minimal_condition_set(tmp_path)
        report = replay_condition_set(tmp_path, envelope)
        assert [
            claim.claim_path
            for claim in report.unchecked_semantic_claims
            if claim.gap is SemanticGap.NO_SUPPORT_OFFERED
        ] == []
