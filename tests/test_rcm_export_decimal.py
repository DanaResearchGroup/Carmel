"""Exact source comparisons before T3's float representation."""

import re
import xml.etree.ElementTree as ET

import pytest
import yaml
from pydantic import ValidationError

from carmel.schemas.datasets import MeasuredValue, UncertaintyBasis
from carmel.services import respecth_series, units
from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason, parse_idt_record
from carmel.services.dataset_store import canonical_decimal
from carmel.services.respecth import RespecthRefusal, replay_idt_record
from carmel.services.respecth import parse_idt_record as parse_respecth
from carmel.services.respecth_archive import load_manifest
from carmel.services.t3_export import export_idt
from tests.test_rcm_export import chemked_history, respecth_record


def uncertainty_record(lower, upper, *, basis=UncertaintyBasis.RELATIVE, idt="0.01", lower_unit=None, upper_unit=None):
    record, _ = respecth_record()
    series = record.envelope.series[0]
    point = series.points[0]
    observation = point.observations[0]
    uncertainty = observation.uncertainty
    template = uncertainty.upper if basis is UncertaintyBasis.RELATIVE else observation.value
    default_unit = "1" if basis is UncertaintyBasis.RELATIVE else "s"

    def measured(template, n, unit):
        fields = template.model_dump(mode="json")
        fields.update(
            raw_text=n,
            canonical_decimal_value=canonical_decimal(n),
            unit_raw=unit,
            unit_normalized=units.normalize_unit(template.quantity_kind, unit, table=units.TABLE_V5),
        )
        return MeasuredValue.model_validate(fields)

    fields = record.model_dump(mode="json")
    points = fields["envelope"]["series"][0]["points"]
    del points[1:]
    fields["rcm_states"] = fields["rcm_states"][:1]
    observation_fields = points[0]["observations"][0]
    observation_fields["value"] = measured(observation.value, idt, "s").model_dump(mode="json")
    observation_fields["uncertainty"].update(
        lower=measured(template, lower, lower_unit or default_unit).model_dump(mode="json"),
        upper=measured(template, upper, upper_unit or default_unit).model_dump(mode="json"),
        basis=basis.value,
    )
    return type(record).model_validate(fields)


@pytest.mark.parametrize(
    "lower,upper,idt",
    [
        ("0.1", "0.10000000000000001", "0.01"),
        ("1E-999", "2E-999", "0.01"),
        ("1E-999", "1E-999", "0.01"),
        ("1E-300", "1E-300", "1E-100"),
        ("1E+308", "1E+308", "10"),
    ],
)
def test_uncertainty_refuses_collapsed_asymmetry_and_unrepresentable_bounds(lower, upper, idt):
    pytest.importorskip("rdkit")
    payload, report = export_idt([uncertainty_record(lower, upper, idt=idt)])
    assert payload["points"] == []
    assert report["refused"] == {"t3_constraint": 1}


@pytest.mark.parametrize(
    "basis,lower,upper,unit",
    [
        (UncertaintyBasis.RELATIVE, "10", "0.1", "%"),
        (UncertaintyBasis.ABSOLUTE, "1", "0.001", "ms"),
    ],
)
def test_exactly_equal_si_bounds_remain_exportable(basis, lower, upper, unit):
    pytest.importorskip("rdkit")
    payload, report = export_idt([uncertainty_record(lower, upper, basis=basis, lower_unit=unit)])
    assert report["exported"] == 1 and not report["refused"]
    assert payload["points"][0]["uncertainty"]["value"] == pytest.approx(0.001)


def argon_record(zero):
    original, raw = respecth_record()
    root = ET.fromstring(raw)
    mixture = root.find("commonProperties/property")
    for child in list(mixture):
        mixture.remove(child)
    for name, smiles, fraction in (("AR", "[Ar]", "1"), ("CH4", "C", zero)):
        component = ET.SubElement(mixture, "component")
        ET.SubElement(component, "speciesLink", preferredKey=name, SMILES=smiles)
        ET.SubElement(component, "amount", units="mole fraction").text = fraction
    group, history = root.findall("dataGroup")
    for point in group.findall("dataPoint")[1:]:
        group.remove(point)
    for point in history.findall("dataPoint"):
        history.remove(point)
    for time, volume in (("0", "1"), ("0.01", "0.5"), ("0.02", "0.6")):
        point = ET.SubElement(history, "dataPoint")
        ET.SubElement(point, "x4").text = time
        ET.SubElement(point, "x5").text = volume
    raw = ET.tostring(root)
    pin = next(a for a in load_manifest().archives if a.name == original.archive.archive_name)
    return parse_respecth(raw, pin, original.archive.member_path), raw


@pytest.mark.parametrize("zero", ["0", "-0", "0.000"])
def test_exact_zero_unsupported_species_is_omitted_from_thermo_only(zero):
    record, raw = argon_record(zero)
    assert replay_idt_record(record, raw).verified
    assert {c.species_raw_name for c in record.envelope.composition.components} == {"AR", "CH4"}


@pytest.mark.parametrize("zero", ["0", "-0", "0.000"])
def test_zero_unsupported_species_exports_with_optional_labels(zero):
    pytest.importorskip("rdkit")
    record, _ = argon_record(zero)
    for derived in (False, True):
        payload, report = export_idt([record], include_derived_labels=derived)
        assert report["exported"] == 1
        assert payload["points"][0]["composition"] == [{"smiles": "[Ar]", "mole_fraction": 1.0}]


def test_nonzero_unsupported_species_that_underflows_is_not_treated_as_exact_zero():
    with pytest.raises(RespecthRefusal, match="outside the seven"):
        argon_record("1E-999")


@pytest.mark.parametrize("fractions", [("1.00000000000000001", "0", "0"), ("0.2", "0.80000100000000001", "0")])
def test_composition_bounds_and_sum_tolerance_use_exact_fractions(fractions):
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    components = doc["datapoints"][0]["composition"]["species"]
    for component, n in zip(components, fractions, strict=True):
        component["amount"] = [n]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "history.yaml")
    assert export_idt([record])[1]["refused"] == {"invalid_composition": 1}


def test_idt_upper_bound_is_checked_before_float_rounding():
    pytest.importorskip("rdkit")
    record = uncertainty_record("0.1", "0.1", idt="10.00000000000000001")
    assert export_idt([record])[1]["refused"] == {"t3_constraint": 1}


def test_history_duration_is_checked_before_float_rounding():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history(explicit=False))
    row = doc["datapoints"][0]
    row["volume-history"]["time"]["units"] = "s"
    row["volume-history"]["values"] = [[10, "0"], [2, "9"], [3, "10.00000000000000001"]]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "history.yaml")
    assert export_idt([record])[1]["refused"] == {"t3_constraint": 1}


def test_documented_unsupported_unit_examples_are_actually_unsupported():
    examples = re.search(r"cannot bind\n\(([^)]*)\)", respecth_series.__doc__).group(1)
    for name in re.findall(r"``([^`]+)``", examples):
        with pytest.raises(units.UnitError):
            units.normalize_unit(units.QuantityKind.PRESSURE, name, table=units.TABLE_V5)


def test_exact_stated_compression_time_cannot_round_inside_source_history():
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    row["volume-history"]["time"]["units"] = "s"
    row["volume-history"]["values"] = [[10, "0"], [2, "9"], [3, "10"]]
    row["compression-time"] = ["10.00000000000000001 s"]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "history.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


def test_pressure_finite_in_source_but_overflowing_in_si_is_a_typed_refusal():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    doc["apparatus"]["kind"] = "shock tube"
    row = doc["datapoints"][0]
    for field in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(field, None)
    row["pressure"] = ["1E+308 bar"]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "pressure.yaml")
    assert export_idt([record])[1]["refused"] == {"missing_stated_state": 1}


@pytest.mark.parametrize("bound", ["0", "-1E-999"])
def test_uncertainty_schema_refuses_nonpositive_bounds_before_export(bound):
    with pytest.raises(ValidationError, match="strictly positive"):
        uncertainty_record(bound, bound)
