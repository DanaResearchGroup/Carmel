"""Regression checks for the offline RCM end-of-compression estimator."""

import hashlib
import json
from pathlib import Path

import pytest

from carmel.services.rcm_thermo import (
    BURCAT_RETRIEVAL_DATE,
    BURCAT_SOURCE_URL,
    BURCAT_THERM_SHA256,
    GRI30_YAML_SHA256,
    NASA7,
    admissible_temperature_domain,
    isentropic_eoc,
)

ORACLE = json.loads((Path(__file__).parent / "fixtures/rcm/cantera-3.2.0-isentropic.json").read_text())


def _cantera_solution_with_burcat_species(ct, gri30_path: Path):
    gas = ct.Solution(str(gri30_path))
    species = gas.species()
    for name, composition in (("NC5H12", {"C": 5, "H": 12}), ("HE", {"He": 1})):
        ranges, high, low = NASA7[name]
        added = ct.Species(name, composition)
        added.thermo = ct.NasaPoly2(ranges[0], ranges[2], ct.one_atm, [ranges[1], *high, *low])
        species.append(added)
    return ct.Solution(thermo="ideal-gas", species=species)


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["id"])
def test_isentropic_estimate_matches_pinned_cantera_oracle(case) -> None:
    """Independent Cantera outputs are checked even when Cantera is unavailable."""
    assert ORACLE["cantera_version"] == "3.2.0"
    assert ORACLE["gri30_yaml_sha256"] == GRI30_YAML_SHA256
    temperature, pressure = isentropic_eoc(
        case["initial_temperature_k"], case["initial_pressure_pa"], case["volume_ratio"], case["composition"]
    )
    assert temperature == pytest.approx(case["temperature_k"], rel=1e-10, abs=1e-6)
    assert pressure == pytest.approx(case["pressure_pa"], rel=1e-9)


def test_live_cantera_regenerates_and_matches_pinned_oracle() -> None:
    ct = pytest.importorskip("cantera")
    path = Path(ct.__file__).parent / "data/gri30.yaml"
    regenerated = {
        "cantera_version": ct.__version__,
        "gri30_yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "procedure": ORACLE["procedure"],
        "cases": [],
    }
    gas = _cantera_solution_with_burcat_species(ct, path)
    for case in ORACLE["cases"]:
        gas.TPX = case["initial_temperature_k"], case["initial_pressure_pa"], case["composition"]
        entropy, specific_volume = gas.SV
        gas.SV = entropy, specific_volume / case["volume_ratio"]
        fresh = {**case, "temperature_k": gas.T, "pressure_pa": gas.P}
        regenerated["cases"].append(fresh)
        assert fresh["temperature_k"] == pytest.approx(case["temperature_k"], rel=2e-12, abs=1e-8)
        assert fresh["pressure_pa"] == pytest.approx(case["pressure_pa"], rel=2e-12, abs=1e-7)
        # Also compare the offline estimator with the regenerated live oracle.
        temperature, pressure = isentropic_eoc(
            case["initial_temperature_k"], case["initial_pressure_pa"], case["volume_ratio"], case["composition"]
        )
        assert temperature == pytest.approx(gas.T, rel=1e-10, abs=1e-6)
        assert pressure == pytest.approx(gas.P, rel=1e-9)
    assert regenerated["cantera_version"] == ORACLE["cantera_version"]
    assert regenerated["gri30_yaml_sha256"] == ORACLE["gri30_yaml_sha256"]
    assert len(regenerated["cases"]) == len(ORACLE["cases"])


def test_pentane_oracle_spans_the_nasa7_breakpoint() -> None:
    cases = [case for case in ORACLE["cases"] if case["id"].startswith("NC5H12-")]
    assert any(case["temperature_k"] < 1000 for case in cases)
    assert any(case["initial_temperature_k"] < 1000 < case["temperature_k"] for case in cases)


def test_argon_compression_has_analytic_monatomic_limit() -> None:
    # Argon's NASA7 cp/R is exactly 2.5: the entropy solution is analytic.
    t, p = isentropic_eoc(300, 101325, 8, {"Ar": 1})
    assert t == pytest.approx(1200, abs=1e-9)
    assert p == pytest.approx(101325 * 32)
    assert isentropic_eoc(3500, 2e5, 1, {"AR": 1}) == pytest.approx((3500, 2e5))
    assert isentropic_eoc(300, 101325, 8, {"AR": 0.99999}) == pytest.approx((t, p))


@pytest.mark.parametrize(
    "temperature,pressure,ratio,composition,message",
    [
        (0, 1e5, 2, {"N2": 1}, "temperature must"),
        (float("nan"), 1e5, 2, {"N2": 1}, "temperature must"),
        (300, 0, 2, {"N2": 1}, "pressure must"),
        (300, float("inf"), 2, {"N2": 1}, "pressure must"),
        (300, 1e5, 0.5, {"N2": 1}, "volume ratio"),
        (300, 1e5, float("inf"), {"N2": 1}, "volume ratio"),
        (300, 1e5, 2, {}, "thermo unavailable"),
        (300, 1e5, 2, {"CH4": 1}, "thermo unavailable"),
        (300, 1e5, 2, {"H2": -1, "O2": 2}, "thermo unavailable"),
        (300, 1e5, 2, {"N2": float("nan")}, "thermo unavailable"),
        (199, 1e5, 2, {"N2": 1}, "polynomial range"),
        (5001, 1e5, 2, {"N2": 1}, "polynomial range"),
        (300, 1e5, 2, {"AR": 0.5, "Ar": 0.5}, "duplicate"),
        (300, 1e5, 2, {"N2": 0.5}, "normalized"),
        (300, 1e5, 1e20, {"AR": 1}, "exceeds"),
        (300, 1e308, 8, {"AR": 1}, "not finite"),
    ],
)
def test_invalid_or_unsupported_thermo_is_refused(temperature, pressure, ratio, composition, message) -> None:
    with pytest.raises(ValueError, match=message):
        isentropic_eoc(temperature, pressure, ratio, composition)


def test_coefficients_cannot_be_mutated() -> None:
    from carmel.services.rcm_thermo import NASA7

    with pytest.raises(TypeError):
        NASA7["AR"] = (0, (), ())
    with pytest.raises(TypeError):
        NASA7["AR"][1][0] = 0


def test_burcat_pin_and_pentane_helium_coefficients_are_frozen() -> None:
    assert BURCAT_SOURCE_URL == "https://respecth.elte.hu/burcat/THERM.DAT"
    assert BURCAT_RETRIEVAL_DATE == "2026-10-17"
    assert BURCAT_THERM_SHA256 == "1bcc0261e5003a94496c07632a5727e312fc9055a18e61fdb7c2d5bdca8eacaa"
    assert NASA7["NC5H12"].temperature_ranges == (200.0, 1000.0, 6000.0)
    assert NASA7["NC5H12"].high == (
        12.6972155,
        0.0304568131,
        -1.09153473e-05,
        1.7552367e-09,
        -1.04805044e-13,
        -25151.8655,
        -43.0901457,
    )
    assert NASA7["NC5H12"].low == (
        6.54437541,
        -0.00659895449,
        0.000151083807,
        -1.93191011e-07,
        7.60508963e-11,
        -20965.4379,
        1.41151597,
    )
    assert NASA7["HE"].high == NASA7["HE"].low == (2.5, 0, 0, 0, 0, -745.375, 0.928723974)


def test_pentane_rcm_mixture_uses_existing_temperature_intersection() -> None:
    composition = {"NC5H12": 0.01, "O2": 0.21, "N2": 0.7799999999974, "HE": 2.6e-12}
    assert admissible_temperature_domain(composition) == (pytest.approx(295), pytest.approx(3500))
    temperature, pressure = isentropic_eoc(360, 1e5, 10, composition)
    assert 360 < temperature < 3500
    assert pressure == pytest.approx(1e5 * 10 * temperature / 360)
