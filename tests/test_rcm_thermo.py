"""Regression checks for the offline RCM end-of-compression estimator."""

import hashlib
import json
from pathlib import Path

import pytest

from carmel.services.rcm_thermo import GRI30_YAML_SHA256, isentropic_eoc

ORACLE = json.loads((Path(__file__).parent / "fixtures/rcm/cantera-3.2.0-isentropic.json").read_text())


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
    gas = ct.Solution(str(path))
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
