"""Public replay and intake regressions for review round 4."""

import xml.etree.ElementTree as ET
from decimal import Decimal

import pytest

from carmel.services import respecth
from carmel.services.respecth_archive import load_manifest
from tests.test_rcm_export import FIXTURES


def precompression_source(*, species="Ar", minimum_volume="0.9"):
    root = ET.fromstring((FIXTURES / "respecth/x40001039.xml").read_bytes())
    composition = root.find("commonProperties/property")
    for child in list(composition):
        composition.remove(child)
    component = ET.SubElement(composition, "component")
    ET.SubElement(component, "speciesLink", preferredKey=species, SMILES="[Ar]" if species == "Ar" else "C")
    ET.SubElement(component, "amount", units="mole fraction").text = "1"
    points, history = root.findall("dataGroup")
    for point in points.findall("dataPoint")[1:]:
        points.remove(point)
    points.find("dataPoint/x2").text = "350"
    for point in history.findall("dataPoint"):
        history.remove(point)
    for time, volume in (("0", "1"), ("0.01", minimum_volume), ("0.02", "0.95")):
        point = ET.SubElement(history, "dataPoint")
        ET.SubElement(point, "x4").text = time
        ET.SubElement(point, "x5").text = volume
    pin = next(a for a in load_manifest().archives if a.name.startswith("syngas"))
    return ET.tostring(root), pin


@pytest.mark.parametrize(
    "case,reason",
    [
        ("weak", respecth.RespecthRefusalReason.IMPLAUSIBLE_IGNITION_TEMPERATURE),
        ("unsupported", respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE),
    ],
)
def test_respecth_replay_refuses_source_that_fails_precompression_admission(monkeypatch, case, reason):
    raw, pin = precompression_source(
        species="Ar" if case == "weak" else "CH4", minimum_volume="0.9" if case == "weak" else "0.4"
    )
    # Construct a forged record with matching provenance, bypassing admission only
    # during construction. Neither parse nor replay below has a bypass in place.
    with monkeypatch.context() as bypass:
        if case == "weak":
            bypass.setattr(respecth, "MIN_IGNITION_TEMPERATURE_K", Decimal(0))
        else:
            bypass.setattr(respecth, "NASA7", {**respecth.NASA7, "CH4": respecth.NASA7["AR"]})
            bypass.setattr(respecth, "isentropic_eoc", lambda *args: (800.0, 100000.0))
        forged = respecth.parse_idt_record(raw, pin, f"{case}.xml")
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, pin, f"{case}.xml")
    assert caught.value.reason is reason
    replay = respecth.replay_idt_record(forged, raw)
    assert not replay.verified
    assert any(reason.value in finding for finding in replay.findings)
