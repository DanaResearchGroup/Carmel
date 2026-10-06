"""Source-built regression coverage for round-6 history and thermo admission."""

import math
import xml.etree.ElementTree as ET
from decimal import Decimal

import pytest
import yaml

from carmel.services import respecth
from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason, parse_idt_record
from carmel.services.rcm_thermo import admissible_temperature_domain, check_initial_temperature, isentropic_eoc
from carmel.services.t3_export import export_idt
from tests.test_rcm_export import chemked_history
from tests.test_rcm_review_round4 import precompression_source


def argon_source(temperature="297", *, unit="K", minimum_volume="0.2"):
    raw, pin = precompression_source(minimum_volume=minimum_volume)
    root = ET.fromstring(raw)
    group = root.find("dataGroup")
    group.find("property[@name='temperature']").set("units", unit)
    group.find("dataPoint/x2").text = temperature
    root.findall("dataGroup")[1].findall("dataPoint")[-1].find("x5").text = "1"
    return ET.tostring(root), pin


@pytest.mark.parametrize("species", ["N2", "AR"])
@pytest.mark.parametrize("temperature", [295, 297, 299.999])
def test_operator_band_admits_only_nitrogen_and_argon(species, temperature):
    assert isentropic_eoc(temperature, 101325, 1, {species: 1}) == pytest.approx((temperature, 101325))
    with pytest.raises(ValueError, match="polynomial range"):
        isentropic_eoc(294.99, 101325, 1, {species: 1})
    with pytest.raises(ValueError, match="polynomial range"):
        isentropic_eoc(5000.01, 101325, 1, {species: 1})


@pytest.mark.parametrize("species", ["H2", "O2", "CO", "CO2", "H2O"])
def test_other_species_have_no_temperature_tolerance(species):
    for temperature in (199.99, 3500.01):
        with pytest.raises(ValueError, match="polynomial range"):
            isentropic_eoc(temperature, 101325, 1, {species: 1})


@pytest.mark.parametrize("species,upper", [("AR", 5000), ("N2", 5000), ("H2", 3500)])
def test_upper_boundary_cannot_compress_even_when_entropy_rounding_hides_the_change(species, upper):
    with pytest.raises(ValueError, match="compressed temperature exceeds"):
        isentropic_eoc(upper, 101325, math.nextafter(1.0, math.inf), {species: 1})


def test_extrapolation_basis_replays_and_exports_in_both_label_modes():
    raw, pin = argon_source()
    record = respecth.parse_idt_record(raw, pin, "extrapolated.xml")
    assert record.rcm_states[0].eoc_basis == "derived-isentropic;thermo=extrapolated-below-300K"
    assert respecth.replay_idt_record(record, raw).verified
    pytest.importorskip("rdkit")
    for labels in (False, True):
        payload, report = export_idt([record], include_derived_labels=labels)
        assert report["exported"] == 1 and not report["refused"]
        point = payload["points"][0]
        assert ";eoc=derived-isentropic;thermo=extrapolated-below-300K" in point["source"]["record"]
        assert ("temperature" in point) == labels


def test_in_range_state_is_not_marked_extrapolated():
    raw, pin = argon_source("300")
    record = respecth.parse_idt_record(raw, pin, "in-range.xml")
    assert record.rcm_states[0].eoc_basis == "derived-isentropic"


@pytest.mark.parametrize("case", ["missing", "null", "empty", "history-null", "not-list"])
@pytest.mark.parametrize("apparatus", ["rapid compression machine", "shock tube"])
def test_present_chemked_history_requires_a_nonempty_values_list(case, apparatus):
    doc = yaml.safe_load(chemked_history())
    doc["apparatus"]["kind"] = apparatus
    row = doc["datapoints"][0]
    if case == "missing":
        row["volume-history"].pop("values")
    elif case == "history-null":
        row["volume-history"] = None
    else:
        row["volume-history"]["values"] = {"null": None, "empty": [], "not-list": "not samples"}[case]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "invalid-history.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize("case", ["shared-id", "empty-id", "missing-id", "missing", "duplicate", "extra"])
def test_respecth_history_requires_distinct_ids_and_exactly_one_cell_each(case):
    raw, pin = argon_source("350")
    root = ET.fromstring(raw)
    history = root.findall("dataGroup")[1]
    if case == "shared-id":
        history.find("property[@name='time']").set("id", "x5")
    elif case == "empty-id":
        history.find("property[@name='time']").set("id", "")
    elif case == "missing-id":
        history.find("property[@name='time']").attrib.pop("id")
    else:
        point = history.find("dataPoint")
        if case == "missing":
            point.remove(point.find("x4"))
        elif case == "duplicate":
            ET.SubElement(point, "x4").text = "0"
        else:
            ET.SubElement(point, "extra").text = "unmapped"
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(ET.tostring(root), pin, "invalid-cells.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.HISTORY_INVALID


def test_respecth_history_reads_each_role_by_its_declared_id():
    raw, pin = argon_source("350")
    root = ET.fromstring(raw)
    history = root.findall("dataGroup")[1]
    history.find("property[@name='time']").set("id", "x5")
    history.find("property[@name='volume']").set("id", "x4")
    for point in history.findall("dataPoint"):
        time, volume = point.find("x4"), point.find("x5")
        time.tag, volume.tag = "x5", "x4"
    raw = ET.tostring(root)
    record = respecth.parse_idt_record(raw, pin, "swapped.xml")
    assert respecth.replay_idt_record(record, raw).verified
    mapped = record.rcm_conditions.histories[0].history
    assert mapped.times == (0, 0.01, 0.02)
    assert mapped.volumes == (1e-6, 2e-7, 1e-6)


@pytest.mark.parametrize(
    "temperature,unit,minimum_volume",
    [
        ("294.999999999999999999", "K", "0.2"),
        ("21.849999999999999999", "degC", "0.2"),
        ("5000.000000000000000001", "K", "0.9999999999999998"),
    ],
)
def test_respecth_initial_domain_uses_exact_base_unit_decimal(temperature, unit, minimum_volume):
    raw, pin = argon_source(temperature, unit=unit, minimum_volume=minimum_volume)
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, pin, "rounded-boundary.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE
    assert "polynomial range" in caught.value.detail


def test_extrapolation_marker_uses_exact_source_temperature():
    raw, pin = argon_source("299.999999999999999999")
    record = respecth.parse_idt_record(raw, pin, "rounded-marker.xml")
    assert record.rcm_states[0].eoc_basis == "derived-isentropic;thermo=extrapolated-below-300K"
    assert respecth.replay_idt_record(record, raw).verified


@pytest.mark.parametrize("temperature", ["294.999999999999999999", "5000.000000000000000001"])
def test_chemked_derived_export_initial_domain_uses_exact_decimal(temperature):
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history(explicit=False))
    row = doc["datapoints"][0]
    row.pop("compressed-temperature")
    row.pop("compressed-pressure")
    row["temperature"] = [f"{temperature} kelvin"]
    row["composition"]["species"] = [{"species-name": "Ar", "SMILES": "[Ar]", "amount": [1.0]}]
    minimum = "0.2" if temperature.startswith("294") else "0.9999999999999998"
    row["volume-history"]["values"] = [["1", 0], [minimum, 10], ["1", 20]]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "export-boundary.yaml")
    payload, report = export_idt([record], include_derived_labels=True)
    assert payload["points"] == []
    assert report["refused"] == {"rcm_thermo_unavailable": 1}


def test_compression_time_containment_is_checked_before_float_rounding():
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0]["compression-time"] = ["8.000000000000000001 ms"]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "compression-boundary.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


def test_derived_compression_time_uses_the_first_exact_source_minimum():
    doc = yaml.safe_load(chemked_history(explicit=False))
    doc["datapoints"][0]["volume-history"]["values"] = [[10, 0], ["2.000000000000000001", 10], ["2", 20], [3, 30]]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "exact-minimum.yaml")
    history = record.rcm_histories[0]
    assert history.volumes[1] == history.volumes[2]
    assert history.compression_time == 0.02


def test_relative_ignition_fraction_upper_bound_is_exact():
    raw, pin = argon_source("350")
    root = ET.fromstring(raw)
    ignition = root.find("ignitionType")
    ignition.set("type", "relative concentration")
    ignition.set("amount", "1.000000000000000001")
    ignition.set("units", "unitless")
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(ET.tostring(root), pin, "ignition-fraction.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.UNMAPPED_IGNITION_DEFINITION


def test_replay_refuses_exact_relative_ignition_fraction_outside_source_bound(monkeypatch):
    raw, pin = argon_source("350")
    root = ET.fromstring(raw)
    ignition = root.find("ignitionType")
    ignition.set("type", "relative concentration")
    ignition.set("amount", "1.000000000000000001")
    ignition.set("units", "unitless")
    raw = ET.tostring(root)
    canonical = respecth.canonical_decimal
    with monkeypatch.context() as bypass:
        bypass.setattr(
            respecth, "canonical_decimal", lambda value: "1" if value == "1.000000000000000001" else canonical(value)
        )
        forged = respecth.parse_idt_record(raw, pin, "ignition-replay.xml")
    replay = respecth.replay_idt_record(forged, raw)
    assert not replay.verified
    assert any("unmapped_ignition_definition" in finding for finding in replay.findings)


@pytest.mark.parametrize(
    "temperature,minimum_volume",
    [("294.999999999999999999", "0.2"), ("5000.000000000000000001", "0.9999999999999998")],
)
def test_replay_applies_exact_temperature_domain_to_source_bytes(monkeypatch, temperature, minimum_volume):
    raw, pin = argon_source(temperature, minimum_volume=minimum_volume)
    with monkeypatch.context() as bypass:
        bypass.setattr(respecth, "check_initial_temperature", lambda *args, **kwargs: False)
        if temperature.startswith("5000"):
            bypass.setattr(respecth, "isentropic_eoc", lambda *args: (800.0, 100000.0))
        forged = respecth.parse_idt_record(raw, pin, "rounded-replay.xml")
    result = respecth.replay_idt_record(forged, raw)
    assert not result.verified
    assert any("rcm_thermo_unavailable" in finding for finding in result.findings)


def test_extrapolation_basis_is_rederived_during_replay():
    raw, pin = argon_source()
    record = respecth.parse_idt_record(raw, pin, "marker-replay.xml")
    fields = record.model_dump(mode="json")
    fields["rcm_states"][0]["eoc_basis"] = "derived-isentropic"
    unmarked = type(record).model_validate(fields)
    result = respecth.replay_idt_record(unmarked, raw)
    assert not result.verified
    assert "RCM initial states do not re-derive" in result.findings


def test_shared_exact_domain_and_extrapolation_classification():
    assert admissible_temperature_domain({"H2": 1}) == (Decimal(200), Decimal(3500))
    assert admissible_temperature_domain({"N2": 0.5, "AR": 0.5, "H2": 0}) == (Decimal(295), Decimal(5000))
    assert admissible_temperature_domain({"H2": 0.5, "AR": 0.5}) == (Decimal(295), Decimal(3500))
    assert not check_initial_temperature(Decimal(297), {"H2": 1})
    assert check_initial_temperature(Decimal(297), {"AR": 1})
    with pytest.raises(ValueError, match="polynomial range"):
        check_initial_temperature(Decimal("NaN"), {"AR": 1})
    for mixture in ({}, {"AR": 0}, {"CH4": 1}):
        with pytest.raises(ValueError, match="thermo unavailable"):
            admissible_temperature_domain(mixture)
