"""Validation and provenance loading for the frozen n-pentane low-T contract."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from builtins import bytes as ByteString
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self
from xml.etree import ElementTree

from pydantic import Field, model_validator

from carmel.benchmarks.chemkin import ChemkinReaction
from carmel.benchmarks.contract import (
    FrozenModel,
    Identity,
    Member,
    Species,
    read_pinned_file,
    validate_ground_truth_counts,
    validate_mechanism_pair,
    validate_species_inventories,
)
from carmel.services.respecth import RespecthRefusal, parse_idt_record
from carmel.services.respecth_archive import PinnedArchive
from carmel.services.respecth_series import parse_series_record

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TypeId = Literal["G1", "G2", "G3", "G4"]


class Pin(FrozenModel):
    filename: str
    source_url: Annotated[str, Field(pattern=r"^https://")]
    retrieval_date: Annotated[str, Field(pattern=r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}$")]
    sha256: Sha256
    bytes: Annotated[int, Field(gt=0)]
    unique_species: Annotated[int, Field(gt=0)] | None = None
    reaction_entries: Annotated[int, Field(gt=0)] | None = None

    def verify(self, data: ByteString) -> None:
        if len(data) != self.bytes or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"PIN CHANGED: {self.filename}: downloaded bytes differ from the frozen contract")


class ArchivePin(Pin):
    osf_path: str
    osf_file_id: Annotated[str, Field(pattern=r"^[0-9a-f]{24}$")]
    osf_version: Annotated[int, Field(ge=1)]

    def to_pinned_archive(self) -> PinnedArchive:
        return PinnedArchive(
            name=self.filename,
            osf_path=self.osf_path,
            osf_file_id=self.osf_file_id,
            osf_version=self.osf_version,
            size=self.bytes,
            sha256=self.sha256,
        )


class GroundTruth(FrozenModel):
    id: TypeId
    name: str
    scope: Literal["primary"]
    rule: str
    reactant_formula: dict[str, int]
    member_count: Annotated[int, Field(gt=0)]
    resolved_count: Annotated[int, Field(ge=0)]
    member_species_count: Annotated[int, Field(gt=0)]
    resolved_species_count: Annotated[int, Field(ge=0)]
    members: tuple[Member, ...]


class SplitFile(FrozenModel):
    path: str
    sha256: Sha256
    rows: Annotated[int, Field(gt=0)]
    observations: Annotated[int, Field(gt=0)]
    low_temperature_rows: Annotated[int, Field(ge=0)]
    low_temperature_observations: Annotated[int, Field(ge=0)]
    acceptance: Literal["supported", "refused"]
    refusal_reason: str | None

    @model_validator(mode="after")
    def frozen_file(self) -> Self:
        path = PurePosixPath(self.path)
        if path.name != self.path or path.suffix != ".xml" or ".." in path.parts:
            raise ValueError("split paths must be plain XML member names")
        if (self.acceptance == "refused") != bool(self.refusal_reason):
            raise ValueError("refused files require a reason and supported files forbid one")
        if self.low_temperature_rows > self.rows or self.low_temperature_observations > self.observations:
            raise ValueError("low-temperature counts cannot exceed all-temperature counts")
        return self


class Study(FrozenModel):
    id: str
    bibliography: str
    doi: str
    set: Literal["development", "holdout"]
    files: tuple[SplitFile, ...]


class Split(FrozenModel):
    archive: ArchivePin
    studies: tuple[Study, ...]


class Scoring(FrozenModel):
    primary_unit: Literal["reaction-type-once"]
    primary_types: tuple[TypeId, ...]
    recovered: str
    baseline_present: str
    secondary_metric: str
    species_aggregation: str
    zero_and_detection_limit_policy: str
    low_temperature_subset: Literal["600-900 K inclusive; reported separately"]
    requires: Literal["jsr-adapter"]
    refusals: tuple[str, ...]
    locking: str


class Contract(FrozenModel):
    schema_version: Literal[1]
    case: Literal["n-pentane-low-t"]
    mechanisms: dict[Literal["v_old", "v_new"], Pin]
    support_sources: tuple[ArchivePin, ...]
    identity: Identity
    new_species: tuple[Species, ...]
    old_species: tuple[Species, ...]
    ground_truth: tuple[GroundTruth, ...]
    split: Split
    scoring: Scoring
    leakage_limits: tuple[str, ...]

    @model_validator(mode="after")
    def frozen_invariants(self) -> Self:
        validate_mechanism_pair(self.mechanisms)
        if self.support_sources != (self.split.archive,):
            raise ValueError("the split archive must be the sole support source")
        if {item.id for item in self.ground_truth} != {"G1", "G2", "G3", "G4"} or len(self.ground_truth) != 4:
            raise ValueError("exactly the four reviewed n-pentane types are required")
        if set(self.scoring.primary_types) != {"G1", "G2", "G3", "G4"} or len(self.scoring.primary_types) != 4:
            raise ValueError("all and only the four reviewed types are primary")
        expected_old_species = self.mechanisms["v_old"].unique_species
        assert expected_old_species is not None
        new = validate_species_inventories(
            self.new_species, self.old_species, expected_old_species=expected_old_species
        )
        for truth in self.ground_truth:
            if dict(truth.reactant_formula) != {"C": 5, "H": 11, "O": 4}:
                raise ValueError("reactant formula is not frozen for n-pentane")
            labels = validate_ground_truth_counts(
                truth.members,
                (
                    truth.member_count,
                    truth.resolved_count,
                    truth.member_species_count,
                    truth.resolved_species_count,
                ),
                new,
            )
            resolved_species = sum(new[label].status == "RESOLVED" for label in labels)
            if resolved_species / len(labels) < 0.7:
                raise ValueError("NEEDS-INPUT: fewer than 70% of scored member species resolved")
        paths = [item.path for study in self.split.studies for item in study.files]
        if len(paths) != len(set(paths)) or len({study.id for study in self.split.studies}) != len(self.split.studies):
            raise ValueError("split files and whole studies must be disjoint")
        dois = [study.doi for study in self.split.studies]
        if len(dois) != len(set(dois)):
            raise ValueError("a source DOI must not cross study groups")
        refused_holdout_rows = sum(
            item.rows
            for study in self.split.studies
            if study.set == "holdout"
            for item in study.files
            if item.acceptance == "refused"
        )
        if refused_holdout_rows * 3 > 144:
            raise ValueError("NEEDS-INPUT: more than one third of holdout rows are refused")
        totals = split_totals(self.split)
        expected = {
            "development_files": 26,
            "development_rows": 226,
            "development_observations": 226,
            "development_supported_files": 0,
            "holdout_files": 6,
            "holdout_rows": 144,
            "holdout_observations": 2080,
            "holdout_low_temperature_rows": 72,
            "holdout_low_temperature_observations": 1040,
            "holdout_supported_files": 6,
        }
        if totals != expected:
            raise ValueError(f"NEEDS-INPUT: frozen split counts do not reproduce: {totals}")
        return self


def condensed_formula(label: str) -> dict[str, int]:
    """Read elemental composition from the complete NUIG baseline label grammar."""
    if label == "AR":
        return {"Ar": 1}
    if label == "HE":
        return {"He": 1}
    if label == "N2":
        return {"N": 2}

    text = re.sub(r"^(?:IIC|TIC|NC|IC|PC|SC|TC|AC)(?=\d)", "C", label)
    text = re.sub(r"^CC(?=\d)", "C", text)
    if "KET" in text:
        carbon = re.search(r"C(\d+)", text)
        if carbon is None:
            raise ValueError(f"unparsed KET composition: {label}")
        count = int(carbon.group(1))
        return {"C": count, "H": 2 * count, "O": 3}

    extra: Counter[str] = Counter()
    if "Q2" in text:
        text = text.replace("Q2", "")
        extra.update({"H": 2, "O": 4})
    if re.search(r"\d-\d,\dOOH", text):
        text = re.sub(r"\d-\d,\dOOH", "", text)
        extra.update({"H": 2, "O": 4})
    text = re.sub(r"OOH\d-\d", "OOH", text)

    def normalize_hydrogen_locant(match: re.Match[str]) -> str:
        carbon = int(match.group(1) or 1)
        hydrogen_digits = match.group(2)
        while int(hydrogen_digits) > 2 * carbon + 2:
            hydrogen_digits = hydrogen_digits[:-1]
        return f"C{carbon}H{hydrogen_digits}"

    text = re.sub(r"C(\d*)H(\d+)", normalize_hydrogen_locant, text)
    cyclic_ether = re.fullmatch(r"C(\d+)H(\d+)O\d-\d", text)
    if cyclic_ether and int(cyclic_ether.group(2)) <= 2 * int(cyclic_ether.group(1)):
        text = f"C{cyclic_ether.group(1)}H{cyclic_ether.group(2)}O"

    atoms = Counter(extra)
    for match in re.finditer(r"([CHO])(\d*)", text):
        atoms[match.group(1)] += int(match.group(2) or 1)
    if not atoms:
        raise ValueError(f"unparsed species composition: {label}")
    return {element: atoms[element] for element in ("C", "H", "N", "O", "Ar", "He") if atoms[element]}


def formula_key(
    labels: tuple[str, ...], formulas: dict[str, dict[str, int]]
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return an order-independent elemental formula profile for one reaction side."""
    return tuple(sorted(tuple(sorted(formulas[label].items())) for label in labels))


def formula_inventory(labels: list[str], reactions: list[ChemkinReaction]) -> dict[str, dict[str, int]]:
    """Decode every species formula and refuse an unbalanced mechanism reaction."""
    formulas = {label: condensed_formula(label) for label in labels}
    for reaction in reactions:
        left = Counter[str]()
        right = Counter[str]()
        for label in reaction["reactants"]:
            left.update(formulas[label])
        for label in reaction["products"]:
            right.update(formulas[label])
        if left != right:
            raise ValueError(f"mechanism composition does not balance: {reaction['equation']}")
    return formulas


def formula_compatible(
    candidate: ChemkinReaction,
    candidate_formulas: dict[str, dict[str, int]],
    prior: ChemkinReaction,
    prior_formulas: dict[str, dict[str, int]],
) -> bool:
    """Return whether two reactions share a formula profile in either orientation."""
    candidate_profile = (
        formula_key(candidate["reactants"], candidate_formulas),
        formula_key(candidate["products"], candidate_formulas),
    )
    prior_profile = (
        formula_key(prior["reactants"], prior_formulas),
        formula_key(prior["products"], prior_formulas),
    )
    return prior_profile in (candidate_profile, candidate_profile[::-1])


def require_structural_absence(
    candidate: ChemkinReaction,
    candidate_formulas: dict[str, dict[str, int]],
    signature: tuple[tuple[str, ...], ...],
    old_reactions: list[ChemkinReaction],
    old_formulas: dict[str, dict[str, int]],
    old_structures: dict[str, str],
) -> None:
    """Refuse unresolved or structurally matching formula-compatible prior reactions."""
    from carmel.benchmarks.structure import reaction_signature

    for prior in old_reactions:
        if not formula_compatible(candidate, candidate_formulas, prior, old_formulas):
            continue
        prior_names = (*prior["reactants"], *prior["products"])
        if not all(label in old_structures for label in prior_names):
            raise ValueError(f"UNRESOLVED old formula-screen candidate: {prior['equation']}")
        prior_signature = reaction_signature(
            tuple(old_structures[label] for label in prior["reactants"]),
            tuple(old_structures[label] for label in prior["products"]),
        )
        if prior_signature == signature:
            raise ValueError(f"baseline reaction signature already present: {prior['equation']}")


def load_contract(path: Path) -> Contract:
    return Contract.model_validate_json(read_pinned_file(path))


def split_totals(split: Split) -> dict[str, int]:
    counts: Counter[str] = Counter(
        {
            "development_supported_files": 0,
            "holdout_supported_files": 0,
        }
    )
    for study in split.studies:
        counts[f"{study.set}_files"] += len(study.files)
        for item in study.files:
            counts[f"{study.set}_rows"] += item.rows
            counts[f"{study.set}_observations"] += item.observations
            if item.acceptance == "supported":
                counts[f"{study.set}_supported_files"] += 1
            if study.set == "holdout":
                counts["holdout_low_temperature_rows"] += item.low_temperature_rows
                counts["holdout_low_temperature_observations"] += item.low_temperature_observations
    return dict(counts)


def series_counts(record: Any) -> tuple[int, int, int, int]:
    series = record.envelope.series[0]
    constants = {item.axis_id: item.value for item in series.constants}
    low_rows = low_observations = observations = 0
    for point in series.points:
        coordinates = {**constants, **{item.axis_id: item.value for item in point.coordinates}}
        temperature = float(coordinates["temperature"].canonical_decimal_value)
        observations += len(point.observations)
        if 600 <= temperature <= 900:
            low_rows += 1
            low_observations += len(point.observations)
    return len(series.points), observations, low_rows, low_observations


def load_split(contract: Contract, archive_source: Path | bytes) -> dict[str, int]:
    """Reverify archive/member pins and run every frozen member through Carmel's readers."""
    raw_archive = archive_source if isinstance(archive_source, bytes) else read_pinned_file(archive_source)
    contract.split.archive.verify(raw_archive)
    archive = contract.split.archive.to_pinned_archive()
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as source:
        for study in contract.split.studies:
            for item in study.files:
                raw = source.read(item.path)
                if hashlib.sha256(raw).hexdigest() != item.sha256:
                    raise ValueError(f"SPLIT BLOB CHANGED: {item.path}")
                root = ElementTree.fromstring(raw)
                groups = [
                    group
                    for group in root.findall("dataGroup")
                    if any(prop.get("name") in {"ignition delay", "composition"} for prop in group.findall("property"))
                ]
                xml_rows = sum(len(group.findall("dataPoint")) for group in groups)
                if xml_rows != item.rows:
                    raise ValueError(f"NEEDS-INPUT: row count changed: {item.path}")
                parser = parse_idt_record if study.set == "development" else parse_series_record
                try:
                    record = parser(raw, archive, item.path)
                    actual_acceptance = "supported"
                    actual_reason = None
                    actual_counts = series_counts(record)
                except RespecthRefusal as exc:
                    actual_acceptance = "refused"
                    actual_reason = exc.reason.value
                    actual_counts = (item.rows, item.observations, 0, 0)
                if (actual_acceptance, actual_reason) != (item.acceptance, item.refusal_reason):
                    raise ValueError(
                        f"DATA ACCEPTANCE CHANGED: {item.path}: "
                        f"{actual_acceptance}/{actual_reason} != {item.acceptance}/{item.refusal_reason}"
                    )
                if actual_acceptance == "supported" and actual_counts != (
                    item.rows,
                    item.observations,
                    item.low_temperature_rows,
                    item.low_temperature_observations,
                ):
                    raise ValueError(f"NEEDS-INPUT: parsed counts changed: {item.path}: {actual_counts}")
    return split_totals(contract.split)


def write_schema(path: Path) -> None:
    path.write_text(json.dumps(Contract.model_json_schema(), indent=2) + "\n")
