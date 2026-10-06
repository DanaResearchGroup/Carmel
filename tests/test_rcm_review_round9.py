"""A source conversion refusal must preserve valid siblings in a batch."""

import hashlib
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from dataclasses import replace
from decimal import Subnormal, localcontext
from io import BytesIO

import pytest
import yaml

from carmel.services import t3_export
from carmel.services.chemked import parse_idt_record
from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest
from carmel.services.respecth_archive import cached_archive_path, load_manifest
from carmel.services.t3_export import export_idt
from tests.test_rcm_export import chemked_history, respecth_record


def test_export_counts_base_unit_overflow_without_losing_a_sibling_point():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    doc["apparatus"]["kind"] = "shock tube"
    row = doc["datapoints"][0]
    for field in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(field, None)
    doc["datapoints"].append(deepcopy(row))
    # Finite and schema-valid; the atm scale grows the 1,000-digit coefficient.
    row["pressure"] = ["1." + "2" * 999 + " atm"]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "conversion-overflow.yaml")

    payload, report = export_idt([record])

    assert report["exported"] == 1
    assert report["refused"] == {"missing_stated_state": 1}
    assert report["refusals"][0]["datapoint"] == 0
    assert ";datapoint=1;" in payload["points"][0]["source"]["record"]
    assert payload["points"][0]["pressure"] == {"value": 100000.0, "units": "Pa"}


def test_export_counts_conversion_arithmetic_failure_without_losing_a_sibling():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    for field in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(field, None)
    doc["datapoints"].append(deepcopy(row))
    doc["datapoints"][1]["pressure"] = ["1 Pa"]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "arithmetic.yaml")
    # Decimal's process context is an external arithmetic constraint. One atm/bar
    # conversion exceeds it; the sibling's identity conversion remains usable.
    with localcontext() as ctx:
        ctx.Emax = 4
        payload, report = export_idt([record])
    assert report["refused"] == {"missing_stated_state": 1}
    assert len(payload["points"]) == 1
    assert ";datapoint=1;" in payload["points"][0]["source"]["record"]


def test_export_counts_idt_conversion_failure_without_losing_a_sibling():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    for field in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(field, None)
    doc["datapoints"].append(deepcopy(row))
    row["ignition-delay"] = ["1E-1000 ms"]
    record = parse_idt_record(yaml.safe_dump(doc).encode(), "idt-conversion.yaml")
    payload, report = export_idt([record])
    assert report["refused"] == {"t3_constraint": 1}
    assert len(payload["points"]) == 1
    assert ";datapoint=1;" in payload["points"][0]["source"]["record"]


def test_export_counts_uncertainty_conversion_failure_without_losing_a_sibling_record():
    from carmel.services.respecth import parse_idt_record as parse_respecth
    from carmel.services.respecth_archive import load_manifest

    pytest.importorskip("rdkit")
    valid, raw = respecth_record()
    root = ET.fromstring(raw)
    uncertainty = root.find("commonProperties/property[@name='evaluated standard deviation']")
    uncertainty.set("units", "%")
    uncertainty.find("value").text = "1E-999"
    pin = next(a for a in load_manifest().archives if a.name == valid.archive.archive_name)
    invalid = parse_respecth(ET.tostring(root), pin, "uncertainty-conversion.xml")
    payload, report = export_idt([invalid, valid])
    expected_points = len(valid.envelope.series[0].points)
    assert report["refused"] == {"t3_constraint": expected_points}
    assert len(payload["points"]) == expected_points
    assert all("uncertainty-conversion.xml" not in p["source"]["record"] for p in payload["points"])


def test_export_counts_history_conversion_failure_without_losing_a_sibling():
    pytest.importorskip("rdkit")
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    row["composition"] = {"kind": "mole fraction", "species": [{"species-name": "Ar", "SMILES": "[Ar]", "amount": [1]}]}
    row["ignition-delay"] = ["1 s"]
    history = parse_idt_record(yaml.safe_dump(doc).encode(), "history-conversion.yaml")
    for field in ("volume-history", "compressed-temperature", "compressed-pressure", "compression-time"):
        row.pop(field, None)
    row["pressure"] = ["1 Pa"]
    sibling = parse_idt_record(yaml.safe_dump(doc).encode(), "sibling.yaml")
    with localcontext() as ctx:
        ctx.Emin = -2
        ctx.traps[Subnormal] = True
        payload, report = export_idt([history, sibling])
    assert report["refused"] == {"t3_constraint": 1}
    assert len(payload["points"]) == 1
    assert "sibling.yaml" in payload["points"][0]["source"]["record"]


@pytest.mark.parametrize(
    "source,case", [("chemked", "composition"), ("respecth", "temperature"), ("respecth", "history")]
)
def test_intake_counts_bad_numeric_values_without_losing_a_sibling_file(monkeypatch, tmp_path, source, case):
    # Only the bundled-manifest read is substituted. Fetch, hash verification,
    # archive unpacking, parsing and replay all use real files and implementations.
    (tmp_path / "sha256").mkdir()
    if source == "chemked":
        from carmel.services.chemked_query import load_idt_records

        raw = chemked_history()
        doc = yaml.safe_load(raw)
        doc["datapoints"][0]["composition"]["species"][0]["amount"] = ["1E+999"]
        bad = yaml.safe_dump(doc).encode()
        files = []
        for name, data in (("bad.yaml", bad), ("valid.yaml", raw)):
            sha = hashlib.sha256(data).hexdigest()
            cached_archive_path(tmp_path, sha).write_bytes(data)
            files.append(ChemkedFile(name, sha))
        manifest = ChemkedManifest("fixture/repository", "0" * 40, tuple(files))
        monkeypatch.setattr(t3_export, "load_manifest", lambda: manifest)
    else:
        from carmel.services.respecth_query import load_idt_records

        valid, raw = respecth_record()
        root = ET.fromstring(raw)
        if case == "temperature":
            root.remove(root.findall("dataGroup")[1])
            apparatus = root.find("apparatus")
            for child in list(apparatus):
                apparatus.remove(child)
            ET.SubElement(apparatus, "kind").text = "shock tube"
            group = root.find("dataGroup")
            prop = group.find("property[@name='temperature']")
            prop.set("units", "C")
            group.find(f"dataPoint/{prop.get('id')}").text = "900." + "2" * 997
        else:
            root.findall("dataGroup")[1].find("dataPoint/x5").text = "1E-999"
        bad = ET.tostring(root)
        stream = BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            bundle.writestr("bad.xml", bad)
            bundle.writestr("valid.xml", raw)
        blob = stream.getvalue()
        sha = hashlib.sha256(blob).hexdigest()
        cached_archive_path(tmp_path, sha).write_bytes(blob)
        manifest = load_manifest()
        pin = next(a for a in manifest.archives if a.name == valid.archive.archive_name)
        manifest = replace(manifest, archives=(replace(pin, sha256=sha, size=len(blob)),))
        monkeypatch.setattr(t3_export, "respecth_manifest", lambda: manifest)

    records, refused = t3_export.load_records(source=source, cache_root=tmp_path, download=False)
    assert len(records) == 1
    assert refused == {source + ":unmapped_unit": 1}
    loaded = load_idt_records(manifest, cache_root=tmp_path, download=False)
    assert len(loaded.records) == 1
    assert loaded.refusals == {"unmapped_unit": 1}
