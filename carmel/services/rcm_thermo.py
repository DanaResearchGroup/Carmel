"""Offline GRI-Mech 3.0 NASA-7 mixture isentropic RCM label estimates.

Coefficients are from cantera/data/gri30.yaml, GRI-Mech 3.0,
https://combustion.berkeley.edu/gri-mech/version30/text30.html.  They were
extracted from Cantera 3.2.0's gri30.yaml (SHA-256 recorded by release).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

GRI30_YAML_SHA256 = "06650b1e0ee0012f6903d5328b1bb218cb6007d07f8ebe375d18f24811039345"
"""Cantera 3.2.0 ``data/gri30.yaml`` from which the coefficients were extracted."""

GRI30_SOURCE_URL = "https://raw.githubusercontent.com/Cantera/cantera/v3.2.0/data/gri30.yaml"
GRI30_RELEASE = "GRI-Mech 3.0 / Cantera 3.2.0"


class Nasa7(NamedTuple):
    """Immutable published temperature domain and coefficients (a1..a7)."""

    temperature_ranges: tuple[float, float, float]
    high: tuple[float, ...]
    low: tuple[float, ...]


NASA7: Mapping[str, Nasa7] = MappingProxyType(
    {
        "H2": Nasa7(
            (200.0, 1000.0, 3500.0),
            (3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331),
            (2.34433112, 0.00798052075, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238),
        ),
        "O2": Nasa7(
            (200.0, 1000.0, 3500.0),
            (3.28253784, 0.00148308754, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129),
            (3.78245636, -0.00299673416, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573),
        ),
        "N2": Nasa7(
            (300.0, 1000.0, 5000.0),
            (2.92664, 0.0014879768, -5.68476e-07, 1.0097038e-10, -6.753351e-15, -922.7977, 5.980528),
            (3.298677, 0.0014082404, -3.963222e-06, 5.641515e-09, -2.444854e-12, -1020.8999, 3.950372),
        ),
        "AR": Nasa7(
            (300.0, 1000.0, 5000.0),
            (2.5, 0, 0, 0, 0, -745.375, 4.366),
            (2.5, 0, 0, 0, 0, -745.375, 4.366),
        ),
        "CO": Nasa7(
            (200.0, 1000.0, 3500.0),
            (2.71518561, 0.00206252743, -9.98825771e-07, 2.30053008e-10, -2.03647716e-14, -14151.8724, 7.81868772),
            (3.57953347, -0.00061035368, 1.01681433e-06, 9.07005884e-10, -9.04424499e-13, -14344.086, 3.50840928),
        ),
        "CO2": Nasa7(
            (200.0, 1000.0, 3500.0),
            (3.85746029, 0.00441437026, -2.21481404e-06, 5.23490188e-10, -4.72084164e-14, -48759.166, 2.27163806),
            (2.35677352, 0.00898459677, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222),
        ),
        "H2O": Nasa7(
            (200.0, 1000.0, 3500.0),
            (3.03399249, 0.00217691804, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14, -30004.2971, 4.9667701),
            (4.19864056, -0.0020364341, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -30293.7267, -0.849032208),
        ),
    }
)


def _s_r(key: str, t: float) -> float:
    ranges, high, low = NASA7[key]
    a = high if t > ranges[1] else low
    return a[0] * math.log(t) + a[1] * t + a[2] * t * t / 2 + a[3] * t**3 / 3 + a[4] * t**4 / 4 + a[6]


def isentropic_eoc(
    temperature_k: float, pressure_pa: float, volume_ratio: float, composition: Mapping[str, float]
) -> tuple[float, float]:
    """Return ideal-gas constant-entropy T,P after V1/V2 compression."""
    if not (math.isfinite(temperature_k) and temperature_k > 0):
        raise ValueError("initial temperature must be finite and positive")
    if not (math.isfinite(pressure_pa) and pressure_pa > 0):
        raise ValueError("initial pressure must be finite and positive")
    if not math.isfinite(volume_ratio) or volume_ratio < 1:
        raise ValueError("volume ratio must be at least one")
    if not composition or any(
        k.upper() not in NASA7 or not math.isfinite(value) or value < 0 for k, value in composition.items()
    ):
        raise ValueError("thermo unavailable")
    x = {k.upper(): v for k, v in composition.items()}
    if len(x) != len(composition):
        raise ValueError("duplicate thermo species aliases")
    if not math.isclose(sum(x.values()), 1.0, rel_tol=0.0, abs_tol=0.005):
        raise ValueError("composition must be normalized")
    # Curated source fractions are sometimes independently rounded to three or five
    # decimal places. Normalize their stated ratios for thermodynamics, exactly
    # as Cantera normalizes its TPX input. Export still preserves/refuses the
    # printed fractions under the downstream schema's stricter tolerance.
    total = sum(x.values())
    x = {key: value / total for key, value in x.items() if value > 0}
    minimum = max(NASA7[key].temperature_ranges[0] for key in x)
    maximum = min(NASA7[key].temperature_ranges[2] for key in x)

    def entropy_at_volume(t: float) -> float:
        if not minimum <= t <= maximum:
            raise ValueError(f"temperature outside the mixture's {minimum}..{maximum} K polynomial range")
        return sum(v * _s_r(k, t) for k, v in x.items()) - math.log(t)

    initial_entropy = entropy_at_volume(temperature_k)
    # For an ideal mixture at fixed composition, S/R = sum(x_i s_i^0/R)
    # - ln(P) + a composition-only mixing term.  Combining P V = n R T
    # at fixed n gives the root condition below.  In particular, this is
    # not the constant-gamma shortcut: the NASA polynomials are evaluated
    # at every trial temperature.
    target = initial_entropy + math.log(volume_ratio)
    lo, hi = temperature_k, maximum
    if entropy_at_volume(hi) < target:
        raise ValueError("compressed temperature exceeds the supported polynomial range")
    for _ in range(100):
        mid = (lo + hi) / 2
        value = entropy_at_volume(mid)
        if value < target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    entropy_at_volume(t)
    pressure = pressure_pa * (volume_ratio * (t / temperature_k))
    if not math.isfinite(pressure):
        raise ValueError("derived pressure is not finite")
    return t, pressure
