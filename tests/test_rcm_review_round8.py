"""Source admission must preserve exact decisions and every declared RCM field."""

import pytest
import yaml

from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason, parse_idt_record
from tests.test_rcm_export import chemked_history


@pytest.mark.parametrize("field", ["compressed-temperature", "compressed-pressure", "compression-time", "paired"])
@pytest.mark.parametrize("apparatus", ["rapid compression machine", "shock tube"])
def test_orphan_rcm_quantities_require_a_history(field, apparatus):
    doc = yaml.safe_load(chemked_history())
    doc["apparatus"]["kind"] = apparatus
    row = doc["datapoints"][0]
    original = dict(row)
    for key in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(key, None)
    for key in ("compressed-temperature", "compressed-pressure") if field == "paired" else (field,):
        row[key] = original[key]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "orphan-state.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize("field", ["compressed-temperature", "compressed-pressure", "compression-time"])
@pytest.mark.parametrize("first", [None, 1, 1.5, False, "", "12", "12 "])
def test_optional_history_quantity_requires_a_value_unit_string(field, first):
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0][field] = [first]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "optional-quantity.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize("time_unit,left,right", [("s", "1", "2"), ("ms", "1000", "2000")])
def test_eoc_between_samples_is_selected_in_exact_base_units(time_unit, left, right):
    from tests.test_rcm_review_round7 import history_source

    doc = yaml.safe_load(history_source([(0, 100), (left, 1), (right, "1E18")], "1.00000000000000001"))
    doc["datapoints"][0]["volume-history"]["time"]["units"] = time_unit
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "decimal-eoc.yaml")
    # Δt = 1E-17 s, so V_eoc = 1 + 1E-17 * (1E18 - 1) ≈ 11 m3.
    assert record.rcm_histories[0].volume_ratio == pytest.approx(100 / 11, rel=1e-15)


def test_exact_compression_is_not_rejected_when_volumes_round_to_the_same_float():
    from tests.test_rcm_review_round7 import history_source

    raw = history_source([(0, "1.00000000000000001"), (1, "1"), (2, "2")], 1)
    record = parse_idt_record(raw, "sub-ulp-compression.yaml")
    assert record.rcm_histories[0].compression_time == 1


def test_unquoted_yaml_source_numbers_keep_all_decimal_digits():
    from decimal import Decimal

    from tests.test_rcm_review_round7 import history_source

    raw = history_source([(0, "1.00000000000000001"), (1, "1"), (2, "2")], 1)
    raw = raw.replace(b"'1.00000000000000001'", b"1.00000000000000001")
    record = parse_idt_record(raw, "unquoted-source.yaml")
    assert record.rcm_histories[0].volume_ratio_decimal == Decimal("1.00000000000000001")


def test_respecth_mixture_normalization_does_not_round_inside_its_tolerance():
    import xml.etree.ElementTree as ET

    from carmel.services import respecth
    from tests.test_rcm_review_round4 import precompression_source

    raw, pin = precompression_source(minimum_volume="0.2")
    root = ET.fromstring(raw)
    mixture = root.find("commonProperties/property")
    mixture.find("component/amount").text = "0.5"
    component = ET.SubElement(mixture, "component")
    ET.SubElement(component, "speciesLink", preferredKey="N2", SMILES="N#N")
    ET.SubElement(component, "amount", units="mole fraction").text = "0.50500000000000001"
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(ET.tostring(root), pin, "mixture-tolerance.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE


def test_respecth_upper_endpoint_refuses_exact_sub_ulp_compression():
    import xml.etree.ElementTree as ET

    from carmel.services import respecth
    from tests.test_rcm_review_round6 import argon_source

    raw, pin = argon_source("5000", minimum_volume="1")
    root = ET.fromstring(raw)
    history = root.findall("dataGroup")[1]
    cells = history.findall("dataPoint/x5")
    for cell, value in zip(cells, ("1.00000000000000001", "1", "1.1"), strict=True):
        cell.text = value
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(ET.tostring(root), pin, "upper-sub-ulp.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE


def test_export_upper_endpoint_refuses_exact_sub_ulp_compression():
    from carmel.services.t3_export import export_idt
    from tests.test_rcm_review_round7 import history_source

    pytest.importorskip("rdkit")
    doc = yaml.safe_load(history_source([(0, "1.00000000000000001"), (1, "1"), (2, "1.1")], 1))
    row = doc["datapoints"][0]
    row.pop("compressed-temperature")
    row.pop("compressed-pressure")
    row["temperature"] = ["5000 K"]
    row["composition"]["species"] = [{"species-name": "Ar", "SMILES": "[Ar]", "amount": [1]}]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "upper-sub-ulp.yaml")
    assert export_idt([record], include_derived_labels=True)[1]["refused"] == {"rcm_thermo_unavailable": 1}


@pytest.mark.parametrize("case", ["non-rcm", "time-unit", "volume-unit", "time-value", "time-order"])
def test_respecth_declared_trace_fields_cannot_skip_validation(case):
    import xml.etree.ElementTree as ET

    from carmel.services import respecth
    from tests.test_rcm_review_round6 import argon_source

    raw, pin = argon_source("800", minimum_volume="2")
    root = ET.fromstring(raw)
    history = root.findall("dataGroup")[1]
    # A valid post-compression, expanding trace formerly skipped time/unit admission.
    history.findall("dataPoint")[-1].find("x5").text = "3"
    if case == "non-rcm":
        apparatus = root.find("apparatus")
        for child in list(apparatus):
            apparatus.remove(child)
        ET.SubElement(apparatus, "kind").text = "shock tube"
    elif case.endswith("-unit"):
        history.find(f"property[@name='{case.split('-')[0]}']").set("units", "unmapped")
    elif case == "time-value":
        history.findall("dataPoint")[1].find("x4").text = "garbage"
    else:
        history.findall("dataPoint")[1].find("x4").text = "-1"
    with pytest.raises(respecth.RespecthRefusal):
        respecth.parse_idt_record(ET.tostring(root), pin, "declared-trace.xml")


def test_export_slope_admission_uses_exact_source_volumes():
    from carmel.services.t3_export import export_idt
    from tests.test_rcm_review_round7 import history_source

    pytest.importorskip("rdkit")
    raw = history_source([(0, "1E308"), ("1E-308", "9.99999999999999999E307"), ("2E-308", "1E308")], "1E-308")
    record = parse_idt_record(raw, "sub-ulp-volume-slope.yaml")
    # Source slope ≈ 1E599; binary volumes collapse, hiding it as zero.
    assert export_idt([record])[1]["refused"] == {"t3_constraint": 1}


@pytest.mark.parametrize("field,value", [("amount", "0.5"), ("units", "unitless")])
def test_chemked_ignition_definition_refuses_an_undefined_declared_amount(field, value):
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    ignition = row.get("ignition-type") or dict(doc["common-properties"]["ignition-type"])
    row["ignition-type"] = {**ignition, field: value}
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "undefined-ignition-amount.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION


def test_exact_mixture_tolerance_endpoint_is_admitted_before_solver_rounding():
    import xml.etree.ElementTree as ET

    from carmel.services import respecth
    from tests.test_rcm_review_round4 import precompression_source

    raw, pin = precompression_source(minimum_volume="0.2")
    root = ET.fromstring(raw)
    mixture = root.find("commonProperties/property")
    mixture.find("component/amount").text = "0.1"
    component = ET.SubElement(mixture, "component")
    ET.SubElement(component, "speciesLink", preferredKey="N2", SMILES="N#N")
    ET.SubElement(component, "amount", units="mole fraction").text = "0.905"
    record = respecth.parse_idt_record(ET.tostring(root), pin, "mixture-endpoint.xml")
    assert record.rcm_states[0].eoc_basis == "derived-isentropic"
