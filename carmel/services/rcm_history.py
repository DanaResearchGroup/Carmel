"""Immutable source-grounded RCM histories and their state semantics."""

from __future__ import annotations

import math
from bisect import bisect_left
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from enum import StrEnum
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from carmel.schemas.datasets import MeasuredValue
from carmel.services import units
from carmel.services.units import QuantityKind


class HistoryReason(StrEnum):
    NONMONOTONE_TIME = "history_nonmonotone_time"
    NONPOSITIVE_VOLUME = "history_nonpositive_volume"
    NO_COMPRESSION = "history_no_compression"
    MISSING_INITIAL_STATE = "history_missing_initial_state"
    INVALID_HISTORY = "history_invalid"


class HistoryRefusal(ValueError):
    def __init__(self, reason: HistoryReason, detail: str):
        self.reason = reason
        super().__init__(f"{reason.value}: {detail}")


def in_si_decimal(value: MeasuredValue) -> Decimal:
    """Convert under the bound table policy before any float representation."""
    table = units.table_for_sha(value.conversion_table_sha256)
    return Decimal(
        units.convert(
            value.canonical_decimal_value,
            quantity=value.quantity_kind,
            from_unit=value.unit_normalized,
            to_unit=table.base_unit(value.quantity_kind),
            table=table,
        ).exact
    )


def in_si(value: MeasuredValue) -> float:
    return float(in_si_decimal(value))


def sample_decimals(
    time: tuple[MeasuredValue, ...], volume: tuple[MeasuredValue, ...]
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    """Validate every trace, including a post-compression expanding trace."""
    if len(time) != len(volume) or len(time) < 2:
        raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "equal-length samples with at least two points required")
    if any(value.quantity_kind is not QuantityKind.TIME for value in time) or any(
        value.quantity_kind is not QuantityKind.VOLUME for value in volume
    ):
        raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "time/volume quantity roles do not match")
    times = tuple(in_si_decimal(value) for value in time)
    volumes = tuple(in_si_decimal(value) for value in volume)
    if any(value <= 0 for value in volumes):
        raise HistoryRefusal(HistoryReason.NONPOSITIVE_VOLUME, "volumes must be positive")
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise HistoryRefusal(HistoryReason.NONMONOTONE_TIME, "times must strictly increase")
    return times, volumes


class RcmHistory(BaseModel):
    """All source samples, with explicit or first-minimum compression time.

    Times keep the source's own coordinates, never rebased; the first sample
    need not be zero. ``compression_time`` is in seconds on that same axis.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    time: tuple[MeasuredValue, ...] = Field(min_length=2)
    volume: tuple[MeasuredValue, ...] = Field(min_length=2)
    compression_time_stated: MeasuredValue | None = None

    @cached_property
    def times(self) -> tuple[float, ...]:
        return tuple(in_si(value) for value in self.time)

    @cached_property
    def volumes(self) -> tuple[float, ...]:
        return tuple(in_si(value) for value in self.volume)

    @cached_property
    def compression_time_decimal(self) -> Decimal:
        if self.compression_time_stated is not None:
            return in_si_decimal(self.compression_time_stated)
        exact_volumes = tuple(in_si_decimal(value) for value in self.volume)
        return in_si_decimal(self.time[exact_volumes.index(min(exact_volumes))])

    @cached_property
    def compression_time(self) -> float:
        return float(self.compression_time_decimal)

    @property
    def compression_time_derived(self) -> bool:
        return self.compression_time_stated is None

    @cached_property
    def end_volume_decimal(self) -> Decimal:
        times = tuple(in_si_decimal(value) for value in self.time)
        volumes = tuple(in_si_decimal(value) for value in self.volume)
        eoc = self.compression_time_decimal
        # _physical checks exact ordering/containment before interval selection.
        index = bisect_left(times, eoc)
        if times[index] == eoc:
            end_volume = volumes[index]
        else:
            with localcontext() as ctx:
                ctx.prec = units._CONVERT_PRECISION
                ctx.rounding = ROUND_HALF_EVEN
                fraction = (eoc - times[index - 1]) / (times[index] - times[index - 1])
                end_volume = (1 - fraction) * volumes[index - 1] + fraction * volumes[index]
        if not end_volume.is_finite() or end_volume <= 0:
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "end volume must be finite and positive")
        return end_volume

    @cached_property
    def volume_ratio_decimal(self) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = units._CONVERT_PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            ratio = in_si_decimal(self.volume[0]) / self.end_volume_decimal
        if not ratio.is_finite() or ratio <= 0:
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "volume ratio must be finite and positive")
        return ratio

    @cached_property
    def volume_ratio(self) -> float:
        # Float checks concern representability only, after exact source decisions.
        end_volume = float(self.end_volume_decimal)
        if not math.isfinite(end_volume) or end_volume == 0:
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "end volume must be finite and positive")
        ratio = float(self.volume_ratio_decimal)
        if not math.isfinite(ratio) or ratio == 0:
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "volume ratio must be finite and positive")
        return ratio

    @model_validator(mode="after")
    def _physical(self) -> RcmHistory:
        exact_times, exact_volumes = sample_decimals(self.time, self.volume)
        if (
            self.compression_time_stated is not None
            and self.compression_time_stated.quantity_kind is not QuantityKind.TIME
        ):
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "compression time must have time units")
        if min(exact_volumes) >= exact_volumes[0]:
            raise HistoryRefusal(HistoryReason.NO_COMPRESSION, "history has no compression phase")
        if self.compression_time_stated is not None and not (
            exact_times[0] <= in_si_decimal(self.compression_time_stated) <= exact_times[-1]
        ):
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "compression time is outside the source history")
        # Also require a usable float representation: distinct samples must not
        # collapse and positive source volumes must not underflow to zero.
        times, volumes = self.times, self.volumes
        if len(times) != len(volumes) or not all(math.isfinite(v) for v in times + volumes):
            raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "finite equal-length samples required")
        if any(v == 0 for v in volumes):
            raise HistoryRefusal(HistoryReason.NONPOSITIVE_VOLUME, "volumes must be positive")
        if len(set(times)) != len(times):
            raise HistoryRefusal(
                HistoryReason.INVALID_HISTORY, "distinct source times collapse in float representation"
            )
        eoc = self.compression_time
        if not math.isfinite(eoc) or self.volume_ratio_decimal <= 1:
            raise HistoryRefusal(
                HistoryReason.INVALID_HISTORY, "compression time must describe compression within the samples"
            )
        _ = self.volume_ratio  # Validate the solver representation after exact admission.
        return self


class RcmState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    initial_temperature: MeasuredValue
    initial_pressure: MeasuredValue
    temperature: MeasuredValue | None = None
    pressure: MeasuredValue | None = None
    eoc_basis: Literal["stated", "derived-isentropic", "derived-isentropic;thermo=extrapolated-below-300K"] = "stated"

    @model_validator(mode="after")
    def _state_roles(self) -> RcmState:
        for value, role in (
            (self.initial_temperature, QuantityKind.TEMPERATURE),
            (self.initial_pressure, QuantityKind.PRESSURE),
            (self.temperature, QuantityKind.TEMPERATURE),
            (self.pressure, QuantityKind.PRESSURE),
        ):
            if value is not None and (
                value.quantity_kind is not role
                or in_si_decimal(value) <= 0
                or not math.isfinite(in_si(value))
                or in_si(value) == 0
            ):
                raise HistoryRefusal(
                    HistoryReason.MISSING_INITIAL_STATE, "finite positive state with correct quantity roles required"
                )
        if (self.temperature is not None and self.pressure is not None) != (self.eoc_basis == "stated"):
            raise ValueError("stated labels require both T/P; derived labels must not carry stated T/P")
        if (self.temperature is None) != (self.pressure is None):
            raise ValueError("end-of-compression temperature and pressure must be paired")
        return self


def history_refusal(error: ValidationError) -> HistoryRefusal:
    for entry in error.errors():
        cause = entry.get("ctx", {}).get("error")
        if isinstance(cause, HistoryRefusal):
            return cause
    return HistoryRefusal(HistoryReason.INVALID_HISTORY, str(error))
