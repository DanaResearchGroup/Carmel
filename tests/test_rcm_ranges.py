"""NASA7 admission must use each material species' published domain."""

import pytest

from carmel.services.rcm_thermo import NASA7, isentropic_eoc


@pytest.mark.parametrize("species", ["N2", "AR"])
@pytest.mark.parametrize("temperature", [250, 299.99, 5000.01])
def test_species_initial_temperature_outside_its_domain_is_refused(species, temperature):
    with pytest.raises(ValueError, match="polynomial range"):
        isentropic_eoc(temperature, 101325, 1, {species: 1})


@pytest.mark.parametrize("species", ["N2", "AR"])
@pytest.mark.parametrize("temperature", [300, 4500, 5000])
def test_species_range_boundaries_and_above_3500_are_supported(species, temperature):
    assert isentropic_eoc(temperature, 101325, 1, {species: 1}) == pytest.approx((temperature, 101325))


def test_positive_species_define_the_intersection_and_bound_compression():
    # Zero H2 contributes no material and must not reduce argon's 5000 K maximum.
    assert isentropic_eoc(4000, 101325, 1, {"AR": 1, "H2": 0}) == pytest.approx((4000, 101325))
    # H2 limits a mixed gas to 3500 K; N2 limits its minimum to 300 K.
    for temperature in (250, 4000):
        with pytest.raises(ValueError, match="polynomial range"):
            isentropic_eoc(temperature, 101325, 1, {"H2": 0.5, "N2": 0.5})
    with pytest.raises(ValueError, match="compressed temperature exceeds"):
        isentropic_eoc(3400, 101325, 2, {"H2": 0.5, "AR": 0.5})
    with pytest.raises(ValueError, match="compressed temperature exceeds"):
        isentropic_eoc(4900, 101325, 2, {"AR": 1})
    temperature, _ = isentropic_eoc(4000, 101325, 1.1, {"AR": 1})
    assert 4000 < temperature < 5000


def test_transcribed_temperature_ranges_are_pinned():
    expected = {
        "H2": (200.0, 1000.0, 3500.0),
        "O2": (200.0, 1000.0, 3500.0),
        "N2": (300.0, 1000.0, 5000.0),
        "AR": (300.0, 1000.0, 5000.0),
        "CO": (200.0, 1000.0, 3500.0),
        "CO2": (200.0, 1000.0, 3500.0),
        "H2O": (200.0, 1000.0, 3500.0),
    }
    assert {key: entry.temperature_ranges for key, entry in NASA7.items()} == expected


def test_transcription_matches_pinned_gri30_when_cantera_is_available():
    import hashlib
    from pathlib import Path

    import yaml

    from carmel.services.rcm_thermo import GRI30_YAML_SHA256

    ct = pytest.importorskip("cantera")
    path = Path(ct.__file__).parent / "data/gri30.yaml"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == GRI30_YAML_SHA256
    species = {item["name"]: item["thermo"] for item in yaml.safe_load(raw)["species"]}
    for name, entry in NASA7.items():
        thermo = species[name]
        assert thermo["model"] == "NASA7"
        assert entry.temperature_ranges == tuple(thermo["temperature-ranges"])
        assert entry.low == tuple(thermo["data"][0])
        assert entry.high == tuple(thermo["data"][1])


def test_nonzero_source_species_cannot_disappear_from_the_range_intersection():
    import xml.etree.ElementTree as ET

    from carmel.services.respecth import RespecthRefusal, RespecthRefusalReason, parse_idt_record
    from tests.test_rcm_review_round4 import precompression_source

    raw, pin = precompression_source()
    root = ET.fromstring(raw)
    root.find("dataGroup/dataPoint/x2").text = "4500"
    composition = root.find("commonProperties/property")
    component = ET.SubElement(composition, "component")
    ET.SubElement(component, "speciesLink", preferredKey="H2", SMILES="[H][H]")
    ET.SubElement(component, "amount", units="mole fraction").text = "1E-999"
    # Ar alone can compress from 4500 K within its 5000 K maximum; any nonzero
    # H2 limits this source mixture to 3500 K, even if its float amount is zero.
    with pytest.raises(RespecthRefusal) as caught:
        parse_idt_record(ET.tostring(root), pin, "tiny-H2.xml")
    assert caught.value.reason is RespecthRefusalReason.RCM_THERMO_UNAVAILABLE
