"""Case-3 pins and temperature bounds on the shared frozen C5 contract."""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self
from xml.etree import ElementTree

from pydantic import Field, model_validator

from carmel.benchmarks.cantera_worker import REVIEWED_NASA_WARNINGS, ReviewedNasaWarning, reviewed_nasa_warning
from carmel.benchmarks.chemkin import NormalizationRewrite
from carmel.benchmarks.contract import FrozenModel, read_pinned_file
from carmel.benchmarks.n_pentane import (
    PentaneContractBase,
    Pin,
    Sha256,
)
from carmel.benchmarks.n_pentane import (
    load_split as load_c5_split,
)


class ThermoPin(Pin):
    reaction_entries: Literal[0] = 0
    nasa_records: Annotated[int, Field(gt=0)]
    covered_mechanism_species: Annotated[int, Field(gt=0)]


class Normalization(FrozenModel):
    raw_sha256: Sha256
    normalized_sha256: Sha256
    rewrites: tuple[NormalizationRewrite, ...]


class Conversion(FrozenModel):
    cantera_version: str
    source_reaction_entries: Annotated[int, Field(gt=0)]
    nonzero_reverse_rates: Annotated[int, Field(ge=0)]
    zero_reverse_rates: Annotated[int, Field(ge=0)]
    converted_reactions: Annotated[int, Field(gt=0)]
    warnings: tuple[str, ...]
    reviewed_warnings: tuple[ReviewedNasaWarning, ...]
    plog_removed_pressures_pa: tuple[float, ...]
    normalization: dict[Literal["mechanism", "thermo"], Normalization]


class Contract(PentaneContractBase):
    case: Literal["iso-pentane-low-t"]
    thermo_sources: dict[Literal["v_old", "v_new"], ThermoPin]
    structure_source: Pin
    companion_page: Pin
    companion_page_evidence: str
    conversion: dict[Literal["v_old", "v_new"], Conversion]
    holdout_max_temperature_k: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    holdout_file_max_temperature_k: dict[str, Annotated[float, Field(gt=0, allow_inf_nan=False)]]

    expected_totals: ClassVar[dict[str, int]] = {
        "development_files": 20,
        "development_rows": 121,
        "development_observations": 121,
        "development_supported_files": 0,
        "holdout_files": 4,
        "holdout_rows": 58,
        "holdout_observations": 812,
        "holdout_low_temperature_rows": 31,
        "holdout_low_temperature_observations": 434,
        "holdout_supported_files": 4,
    }

    @model_validator(mode="after")
    def conversion_invariants(self) -> Self:
        if set(self.thermo_sources) != {"v_old", "v_new"} or set(self.conversion) != {"v_old", "v_new"}:
            raise ValueError("both thermo pins and conversion audits are required")
        paths = {item.path for study in self.split.studies if study.set == "holdout" for item in study.files}
        if set(self.holdout_file_max_temperature_k) != paths:
            raise ValueError("temperature bounds must cover every holdout file")
        if self.holdout_max_temperature_k != max(self.holdout_file_max_temperature_k.values()):
            raise ValueError("holdout maximum temperature disagrees with file bounds")
        for endpoint, audit in self.conversion.items():
            expected_reviews = REVIEWED_NASA_WARNINGS[2:] if endpoint == "v_old" else REVIEWED_NASA_WARNINGS[:2]
            if audit.reviewed_warnings != expected_reviews or len(audit.warnings) != len(expected_reviews):
                raise ValueError("conversion must retain every reviewed warning it relies on")
            if tuple(reviewed_nasa_warning(warning) for warning in audit.warnings) != expected_reviews:
                raise ValueError("unreviewed NASA validation warning")
            mechanism = self.mechanisms[endpoint]
            thermo = self.thermo_sources[endpoint]
            if thermo.covered_mechanism_species != mechanism.unique_species:
                raise ValueError("thermo must cover all mechanism species")
            if audit.source_reaction_entries != mechanism.reaction_entries:
                raise ValueError("pin must count source reaction entries")
            if audit.converted_reactions != audit.source_reaction_entries + audit.nonzero_reverse_rates:
                raise ValueError("converted reaction count must account for explicit reverse rates")
            if set(audit.normalization) != {"mechanism", "thermo"}:
                raise ValueError("normalization must inventory both inputs")
            inputs: tuple[tuple[Literal["mechanism", "thermo"], Pin], ...] = (
                ("mechanism", mechanism),
                ("thermo", thermo),
            )
            for kind, pin in inputs:
                if audit.normalization[kind].raw_sha256 != pin.sha256:
                    raise ValueError("normalization raw hash differs from input pin")
            for review in audit.reviewed_warnings:
                if review.requires_below_tmid and self.holdout_max_temperature_k >= review.tmid_k:
                    raise ValueError("NEEDS-INPUT: holdout maximum must stay below reviewed Tmid")
        return self


def load_contract(path: Path) -> Contract:
    return Contract.model_validate_json(read_pinned_file(path))


def xml_temperature_max(raw: bytes) -> float:
    """Count temperature from the pinned XML, including rows refused by a reader."""
    root = ElementTree.fromstring(raw)
    values: list[float] = []
    for group in root.findall("dataGroup"):
        if not any(p.get("name") in {"composition", "ignition delay"} for p in group.findall("property")):
            continue
        temperatures = [p for p in group.findall("property") if p.get("name") == "temperature"]
        if len(temperatures) != 1 or temperatures[0].get("units") != "K":
            raise ValueError("temperature maximum requires one explicit Kelvin axis")
        axis = temperatures[0].get("id")
        if axis is None:
            raise ValueError("missing temperature axis ID")
        values.extend(float(point.findtext(axis) or "nan") for point in group.findall("dataPoint"))
    if not values or any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("missing or invalid temperature for frozen row")
    return max(values)


def load_split(contract: Contract, archive_source: Path | bytes) -> dict[str, int]:
    """Use the shared reader replay, then verify the temperature-dependent review domain."""
    raw = archive_source if isinstance(archive_source, bytes) else read_pinned_file(archive_source)
    totals = load_c5_split(contract, raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        for path, bound in contract.holdout_file_max_temperature_k.items():
            if xml_temperature_max(source.read(path)) != bound:
                raise ValueError(f"holdout temperature maximum changed: {path}")
    return totals
