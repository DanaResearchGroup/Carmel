"""Offline GRI-Mech 3.0 NASA-7 mixture isentropic RCM label estimates.

Coefficients are from cantera/data/gri30.yaml, GRI-Mech 3.0,
https://combustion.berkeley.edu/gri-mech/version30/text30.html.  They were
extracted from Cantera 3.2.0's gri30.yaml (SHA-256 recorded by release).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from types import MappingProxyType
from typing import Literal, NamedTuple

from carmel.services import units

GRI30_YAML_SHA256 = "06650b1e0ee0012f6903d5328b1bb218cb6007d07f8ebe375d18f24811039345"
"""Cantera 3.2.0 ``data/gri30.yaml`` from which the coefficients were extracted."""

GRI30_SOURCE_URL = "https://raw.githubusercontent.com/Cantera/cantera/v3.2.0/data/gri30.yaml"
GRI30_RELEASE = "GRI-Mech 3.0 / Cantera 3.2.0"

N2_AR_LOW_T_EXTRAPOLATION_K = Decimal("5")
"""Operator-approved low-T tolerance: N2/Ar polynomials are nearly flat over
a few kelvin below 300 K. No other species or upper bound receives a tolerance.
"""

EXTRAPOLATED_EOC_BASIS: Literal["derived-isentropic;thermo=extrapolated-below-300K"] = (
    "derived-isentropic;thermo=extrapolated-below-300K"
)


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


def admissible_temperature_domain(composition: Mapping[str, Decimal | float]) -> tuple[Decimal, Decimal]:
    """Exact Kelvin bounds for the positive mixture, including the N2/Ar ruling."""
    exact = {key: Decimal(str(value)) for key, value in composition.items()}
    if any(not value.is_finite() or value < 0 for value in exact.values()):
        raise ValueError("thermo unavailable")
    positive = {key: value for key, value in exact.items() if value > 0}
    keys = {key.upper() for key in positive}
    if len(keys) != len(positive):
        raise ValueError("duplicate thermo species aliases")
    if not keys or any(key not in NASA7 for key in keys):
        raise ValueError("thermo unavailable")
    lower = max(
        Decimal(str(NASA7[key].temperature_ranges[0]))
        - (N2_AR_LOW_T_EXTRAPOLATION_K if key in {"N2", "AR"} else Decimal(0))
        for key in keys
    )
    upper = min(Decimal(str(NASA7[key].temperature_ranges[2])) for key in keys)
    return lower, upper


def _source_total(composition: Mapping[str, Decimal | float]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = units._CONVERT_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        total = sum((Decimal(str(amount)) for amount in composition.values()), Decimal(0))
        if abs(total - Decimal(1)) > Decimal("0.005"):
            raise ValueError("composition must be normalized")
    return total


def solver_composition(composition: Mapping[str, Decimal]) -> dict[str, float]:
    """Admit and normalize exact source ratios before representing solver inputs."""
    admissible_temperature_domain(composition)
    total = _source_total(composition)
    with localcontext() as ctx:
        ctx.prec = units._CONVERT_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        result = {key: float(amount / total) for key, amount in composition.items() if not amount.is_zero()}
    if any(not math.isfinite(value) or value == 0 for value in result.values()):
        raise ValueError("a nonzero source fraction underflows the thermodynamic mixture's float representation")
    return result


def check_initial_temperature(
    temperature: Decimal,
    composition: Mapping[str, Decimal | float],
    *,
    volume_ratio: Decimal | None = None,
) -> bool:
    """Check exact base-unit admission; return whether the low-T band is used.

    Source callers must use this before representing the temperature as float.
    """
    lower, upper = admissible_temperature_domain(composition)
    _source_total(composition)
    if not temperature.is_finite() or not lower <= temperature <= upper:
        raise ValueError(f"temperature outside the mixture's {lower}..{upper} K polynomial range")
    if volume_ratio is not None and temperature == upper and volume_ratio > 1:
        raise ValueError("compressed temperature exceeds the supported polynomial range")
    nominal_lower = max(
        Decimal(str(NASA7[key.upper()].temperature_ranges[0]))
        for key, amount in composition.items()
        if Decimal(str(amount)) > 0
    )
    return temperature < nominal_lower


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
    _source_total(x)
    # Curated source fractions are sometimes independently rounded to three or five
    # decimal places. Normalize their stated ratios for thermodynamics, exactly
    # as Cantera normalizes its TPX input. Export still preserves/refuses the
    # printed fractions under the downstream schema's stricter tolerance.
    total = sum(x.values())
    x = {key: value / total for key, value in x.items() if value > 0}
    bounds = admissible_temperature_domain(x)
    minimum, maximum = float(bounds[0]), float(bounds[1])

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
    # Any compression at the upper endpoint leaves the domain, even when a
    # sub-ulp entropy change rounds back to the initial floating value.
    if (temperature_k == maximum and volume_ratio > 1) or entropy_at_volume(hi) < target:
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
