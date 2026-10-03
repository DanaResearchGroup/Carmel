"""Source-built regressions for RCM review round 3."""

import xml.etree.ElementTree as ET

import pytest
import yaml

from carmel.services.chemked import parse_idt_record, replay_idt_record
from carmel.services.respecth import RespecthRefusal, RespecthRefusalReason
from carmel.services.respecth import parse_idt_record as parse_respecth
from carmel.services.respecth_archive import load_manifest
from carmel.services.t3_export import export_idt
from tests.test_rcm_export import FIXTURES, chemked_history


def test_weak_respecth_compression_refuses_implausible_derived_temperature():
    root = ET.fromstring((FIXTURES / "respecth/x40001039.xml").read_bytes())
    composition = root.find("commonProperties/property")
    for child in list(composition):
        composition.remove(child)
    component = ET.SubElement(composition, "component")
    ET.SubElement(component, "speciesLink", preferredKey="Ar", SMILES="[Ar]")
    ET.SubElement(component, "amount", units="mole fraction").text = "1"
    points, history = root.findall("dataGroup")
    for point in points.findall("dataPoint")[1:]:
        points.remove(point)
    points.find("dataPoint/x2").text = "350"
    for point in history.findall("dataPoint"):
        history.remove(point)
    for time, volume in (("0", "1"), ("0.01", "0.9"), ("0.02", "0.95")):
        point = ET.SubElement(history, "dataPoint")
        ET.SubElement(point, "x4").text = time
        ET.SubElement(point, "x5").text = volume
    pin = next(a for a in load_manifest().archives if a.name.startswith("syngas"))
    with pytest.raises(RespecthRefusal) as caught:
        parse_respecth(ET.tostring(root), pin, "weak.xml")
    assert caught.value.reason is RespecthRefusalReason.IMPLAUSIBLE_IGNITION_TEMPERATURE
    assert "500 K" in caught.value.detail


def test_chemked_weak_compression_exports_history_but_refuses_implausible_derived_label():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history(explicit=False))
    row = doc["datapoints"][0]
    row.pop("compressed-temperature")
    row.pop("compressed-pressure")
    row["composition"]["species"] = [{"species-name": "Ar", "SMILES": "[Ar]", "amount": [1.0]}]
    row["volume-history"]["values"] = [[1, 0], [0.9, 10], [0.95, 20]]
    raw = yaml.safe_dump(doc).encode()
    record = parse_idt_record(raw, "weak.yaml")
    replay_idt_record(record, raw)
    assert export_idt([record])[1]["unlabelled"] == 1
    payload, report = export_idt([record], include_derived_labels=True)
    assert payload["points"] == []
    assert report["refused"] == {"implausible_ignition_temperature": 1}


@pytest.mark.parametrize("sibling_identity", ["missing", "conflicting"])
def test_chemked_identity_is_scoped_to_source_datapoint(sibling_identity):
    from copy import deepcopy

    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"].append(deepcopy(doc["datapoints"][0]))
    sibling = doc["datapoints"][1]
    sibling["ignition-delay"] = ["2 ms"]
    fuel = sibling["composition"]["species"][0]
    if sibling_identity == "missing":
        fuel.pop("InChI")
    else:
        fuel["SMILES"] = "C"
    raw = yaml.safe_dump(doc).encode()
    record = parse_idt_record(raw, "sibling.yaml")
    replay_idt_record(record, raw)
    # The native parser emits one point per row, without filtering or reordering.
    for index, point in enumerate(record.envelope.series[0].points):
        assert all(c.value.value_ref.locator.path.startswith(f"datapoints[{index}].") for c in point.coordinates)
        assert point.observations[0].value.value_ref.locator.path == f"datapoints[{index}].ignition-delay[0]#value"
    payload, report = export_idt([record])
    assert report["exported"] == 1
    assert report["refused"] == {"no_confident_smiles": 1}
    assert report["refusals"][0]["datapoint"] == 1
    assert ";datapoint=0;" in payload["points"][0]["source"]["record"]


def test_respecth_identities_remain_file_scoped_for_all_points():
    from tests.test_rcm_export import respecth_record

    pytest.importorskip("rdkit")
    record, _ = respecth_record()
    payload, report = export_idt([record])
    assert report["exported"] == len(record.envelope.series[0].points)
    assert not report["refused"]
    assert all(p["composition"] == payload["points"][0]["composition"] for p in payload["points"])


@pytest.mark.parametrize("time_ms,ratio", [(7, 5), (7.1, 10 / 2.1), (8, 10 / 3)])
def test_validated_history_interpolates_at_samples_and_between_them(time_ms, ratio):
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0]["compression-time"] = [f"{time_ms} ms"]
    raw = yaml.safe_dump(doc).encode()
    record = parse_idt_record(raw, "interpolation.yaml")
    replay_idt_record(record, raw)
    assert record.rcm_histories[0].volume_ratio == pytest.approx(ratio)
