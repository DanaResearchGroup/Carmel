"""Validation and provenance loading for the frozen n-heptane low-T contract."""

from __future__ import annotations

import hashlib
import json
from builtins import bytes as ByteString
from collections import Counter
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal, NoReturn, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carmel.services.archive_unpack import _is_absolute_member_name
from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason
from carmel.services.chemked_archive import _MAX_FILE_BYTES

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
TypeId = Literal["G1", "G2", "G3", "G4", "G5", "G6"]

EXPECTED_REACTANT_FORMULAS = MappingProxyType(
    {
        **{type_id: MappingProxyType({"C": 7, "H": 15, "O": 4}) for type_id in ("G1", "G2", "G3", "G4", "G5")},
        "G6": MappingProxyType({"C": 7, "H": 14, "O": 3}),
    }
)


def expected_reactant_formula(type_id: TypeId) -> dict[str, int]:
    return dict(EXPECTED_REACTANT_FORMULAS[type_id])


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def freeze_mappings(self) -> Self:
        for name, value in self.__dict__.items():
            if isinstance(value, dict) and not isinstance(value, ImmutableDict):
                object.__setattr__(self, name, ImmutableDict(value))
        return self


class ImmutableDict(dict[str, Any]):
    """A JSON-object-shaped mapping that cannot be mutated in place."""

    def _reject_mutation(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("mapping is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        self._reject_mutation(key, value)

    def __delitem__(self, key: str) -> None:
        self._reject_mutation(key)

    def clear(self) -> None:
        self._reject_mutation()

    def pop(self, *args: object) -> Any:
        self._reject_mutation(*args)

    def popitem(self) -> tuple[str, Any]:
        self._reject_mutation()

    def setdefault(self, *args: object) -> Any:
        self._reject_mutation(*args)

    def update(self, *args: object, **kwargs: Any) -> None:
        self._reject_mutation(*args, **kwargs)

    def _immutable_ior(self, other: object) -> Self:
        self._reject_mutation(other)

    __ior__ = _immutable_ior


class Pin(FrozenModel):
    filename: str
    source_url: Annotated[str, Field(pattern=r"^https://")]
    retrieval_date: Literal["2026-10-02"]
    sha256: Sha256
    bytes: Annotated[int, Field(gt=0)]
    unique_species: Annotated[int, Field(gt=0)] | None = None
    reaction_entries: Annotated[int, Field(gt=0)] | None = None

    def verify(self, data: ByteString) -> None:
        if len(data) != self.bytes or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"PIN CHANGED: {self.filename}: downloaded bytes differ from the frozen contract")


class Species(FrozenModel):
    label: str
    status: Literal["RESOLVED", "UNRESOLVED"]
    smiles: str | None
    formula: dict[str, Annotated[int, Field(ge=0)]]
    source: str

    @model_validator(mode="after")
    def structure_status(self) -> Self:
        if (self.status == "RESOLVED") != bool(self.smiles):
            raise ValueError("UNRESOLVED species must not carry a guessed structure")
        return self


class Member(FrozenModel):
    equation: str
    line: Annotated[int, Field(gt=0)]
    reactants: tuple[str, ...]
    products: tuple[str, ...]
    absence: Literal["structural-signature-and-exhaustive-formula-screen"]


class GroundTruth(FrozenModel):
    id: TypeId
    name: str
    scope: Literal["primary", "speciation-only"]
    rule: str
    reactant_formula: dict[str, int]
    member_count: Annotated[int, Field(gt=0)]
    resolved_count: Annotated[int, Field(ge=0)]
    member_species_count: Annotated[int, Field(gt=0)]
    resolved_species_count: Annotated[int, Field(ge=0)]
    members: tuple[Member, ...]

    @model_validator(mode="after")
    def frozen_reactant_formula(self) -> Self:
        if dict(self.reactant_formula) != expected_reactant_formula(self.id):
            raise ValueError(f"reactant formula is not frozen for {self.id}")
        return self


class SplitFile(FrozenModel):
    path: str
    git_blob: GitSha
    sha256: Sha256
    points: Annotated[int, Field(gt=0)]
    below_750: Annotated[int, Field(ge=0)]
    from_750_to_900: Annotated[int, Field(ge=0)]
    rcm_ideal_points: tuple[int, ...]

    @model_validator(mode="after")
    def safe_path(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            _is_absolute_member_name(self.path)
            or "\\" in self.path
            or ".." in path.parts
            or not self.path.startswith("n-heptane/")
        ):
            raise ValueError("split paths must stay beneath the ChemKED n-heptane directory")
        if len(set(self.rcm_ideal_points)) != len(self.rcm_ideal_points) or any(
            i < 0 or i >= self.points for i in self.rcm_ideal_points
        ):
            raise ValueError("RCM point indices must be unique and in range")
        if self.below_750 + self.from_750_to_900 > self.points:
            raise ValueError("temperature bins cannot exceed total points")
        return self


class Study(FrozenModel):
    id: str
    doi: str | None
    set: Literal["development", "holdout"]
    files: tuple[SplitFile, ...]


class Split(FrozenModel):
    repository: Literal["pr-omethe-us/ChemKED-database"]
    commit: Literal["606005bfc8f5214b3f0b5ca7300a96a82815c2ae"]
    frozen_at: Literal["2026-10-02T04:33:18.068272+00:00"]
    studies: tuple[Study, ...]


class Scoring(FrozenModel):
    primary_unit: Literal["reaction-type-once"]
    primary_types: tuple[TypeId, ...]
    recovered: str
    baseline_present: str
    g5_merge: Literal["only-if-structural-matcher-cannot-separate-G5-from-G1;declare-before-lock"]
    secondary_idt: str
    secondary_groups: tuple[Literal["shock-tube", "RCM-ideal"], ...]
    improvement: str
    speciation: str
    coverage: str
    refusals: tuple[str, ...]
    locking: str


class Identity(FrozenModel):
    status: Literal["UNVERIFIED"]
    historical_identity_caveat: str
    attempts: tuple[str, ...]
    evidence: tuple[str, ...]


class Contract(FrozenModel):
    schema_version: Literal[1]
    case: Literal["n-heptane-low-t"]
    mechanisms: dict[Literal["v_old", "v_new"], Pin]
    support_sources: tuple[Pin, ...]
    identity: Identity
    new_species: tuple[Species, ...]
    old_species: tuple[Species, ...]
    ground_truth: tuple[GroundTruth, ...]
    split: Split
    scoring: Scoring
    leakage_limits: tuple[str, ...]

    @model_validator(mode="after")
    def frozen_invariants(self) -> Self:
        if set(self.mechanisms) != {"v_old", "v_new"}:
            raise ValueError("both mechanism pins are required")
        if tuple(source.filename for source in self.support_sources) != (
            "i117-glossary.pdf",
            "i117-v31-thermo.txt",
        ):
            raise ValueError("support sources must be glossary then thermo")
        if {g.id for g in self.ground_truth} != {"G1", "G2", "G3", "G4", "G5", "G6"} or len(self.ground_truth) != 6:
            raise ValueError("exactly the six reviewed types are required")
        if set(self.scoring.primary_types) != {"G1", "G2", "G3", "G4", "G5"} or len(self.scoring.primary_types) != 5:
            raise ValueError("G6 must never be scored against ignition")
        new = {s.label: s for s in self.new_species}
        if len(new) != len(self.new_species) or len({s.label for s in self.old_species}) != len(self.old_species):
            raise ValueError("species labels must be unique within each mechanism")
        if len(self.old_species) != self.mechanisms["v_old"].unique_species:
            raise ValueError("absence audit must inventory every old species")
        for gt in self.ground_truth:
            labels = {s for m in gt.members for s in (*m.reactants, *m.products)}
            if labels - new.keys():
                raise ValueError("member references an uninventoried species")
            resolved = sum(all(new[s].status == "RESOLVED" for s in (*m.reactants, *m.products)) for m in gt.members)
            resolved_species = sum(new[s].status == "RESOLVED" for s in labels)
            if (gt.member_count, gt.resolved_count, gt.member_species_count, gt.resolved_species_count) != (
                len(gt.members),
                resolved,
                len(labels),
                resolved_species,
            ):
                raise ValueError("ground-truth counts do not reproduce")
            if gt.scope != ("speciation-only" if gt.id == "G6" else "primary"):
                raise ValueError("G6 is speciation-only; G1-G5 are primary")
            if gt.scope == "primary" and resolved_species / len(labels) < 0.7:
                raise ValueError("NEEDS-INPUT: fewer than 70% of scored member species resolved")
            if len({m.line for m in gt.members}) != len(gt.members):
                raise ValueError("member entry lines must be unique; duplicate kinetics retain separate lines")
        paths = [f.path for study in self.split.studies for f in study.files]
        if len(set(paths)) != len(paths) or len({s.id for s in self.split.studies}) != len(self.split.studies):
            raise ValueError("split files and whole studies must be disjoint")
        dois = [s.doi for s in self.split.studies if s.doi is not None]
        if len(set(dois)) != len(dois):
            raise ValueError("a source DOI must not cross study groups")
        totals = split_totals(self.split)
        if totals != {
            "development_files": 45,
            "development_points": 521,
            "holdout_files": 24,
            "holdout_points": 189,
            "holdout_below_750": 26,
            "holdout_from_750_to_900": 63,
            "holdout_rcm_ideal": 15,
        }:
            raise ValueError(f"NEEDS-INPUT: frozen split counts do not reproduce: {totals}")
        if next((s.set for s in self.split.studies if s.id == "Zhang 2016"), None) != "development":
            raise ValueError("all Zhang2016 data must stay in development")
        return self

    @property
    def glossary_source(self) -> Pin:
        return self.support_sources[0]

    @property
    def thermo_source(self) -> Pin:
        return self.support_sources[1]


def load_contract(path: Path) -> Contract:
    return Contract.model_validate_json(_read_pinned_root_file(path))


def split_totals(split: Split) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for study in split.studies:
        counts[f"{study.set}_files"] += len(study.files)
        for f in study.files:
            counts[f"{study.set}_points"] += f.points
            if study.set == "holdout":
                counts["holdout_below_750"] += f.below_750
                counts["holdout_from_750_to_900"] += f.from_750_to_900
                counts["holdout_rcm_ideal"] += len(f.rcm_ideal_points)
    return dict(counts)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def load_split(
    contract: Contract, root: Path | None = None, *, fetcher: Callable[[SplitFile], bytes] | None = None
) -> dict[str, int]:
    """Verify every blob, point count, temperature bin and RCM flag before using data.

    ``root`` is a ChemKED directory (containing ``n-heptane/``). Otherwise fetch
    through Carmel's pinned manifest/cache. ``fetcher`` permits an explicit fresh
    fetch or an offline fixture; neither mode bypasses byte verification.
    """
    if root is None and fetcher is None:
        from carmel.services.chemked_archive import fetch_file, load_manifest

        manifest = load_manifest()
        manifest_files = {f.path: f for f in manifest.files}
        if manifest.repository != contract.split.repository:
            raise ValueError("Carmel ChemKED repository differs from the frozen split")
        if manifest.commit != contract.split.commit:
            raise ValueError("Carmel ChemKED commit differs from the frozen split")

        def cached_fetch(item: SplitFile) -> bytes:
            pinned = manifest_files[item.path]
            if pinned.sha256 != item.sha256:
                raise ValueError("Carmel ChemKED pin differs from the frozen split")
            return fetch_file(pinned, manifest, Path.home() / ".cache" / "carmel" / "chemked")

        fetcher = cached_fetch
    for study in contract.split.studies:
        for item in study.files:
            if root is not None:
                raw = _read_pinned_root_file(root / item.path)
            else:
                assert fetcher is not None
                raw = fetcher(item)
            if git_blob(raw) != item.git_blob or hashlib.sha256(raw).hexdigest() != item.sha256:
                raise ValueError(f"SPLIT BLOB CHANGED: {item.path}")
            data = yaml.safe_load(raw)
            apparatus = data.get("apparatus") if isinstance(data, dict) else None
            kind = apparatus.get("kind") if isinstance(apparatus, dict) else None
            if kind not in {"shock tube", "rapid compression machine"}:
                raise ChemkedRefusal(ChemkedRefusalReason.UNMAPPED_APPARATUS, f"{item.path}: {apparatus!r}")
            points = data["datapoints"]
            temperatures = []
            ideal = []
            for i, point in enumerate(points):
                value, unit = point["temperature"][0].split()
                if unit not in {"K", "kelvin"} or "ignition-delay" not in point:
                    raise ValueError(f"unscored: unsupported temperature unit or missing IDT: {item.path}:{i}")
                temperatures.append(float(value))
                if data["apparatus"]["kind"] == "rapid compression machine":
                    if "volume-history" in point or any(h["type"] == "volume" for h in point.get("time-histories", [])):
                        raise ValueError("frozen RCM-ideal point unexpectedly carries a volume history")
                    ideal.append(i)
            actual = (
                len(points),
                sum(t < 750 for t in temperatures),
                sum(750 <= t <= 900 for t in temperatures),
                tuple(ideal),
            )
            expected = (item.points, item.below_750, item.from_750_to_900, item.rcm_ideal_points)
            if actual != expected:
                raise ValueError(f"NEEDS-INPUT: point/RCM counts changed: {item.path}: {actual} != {expected}")
    return split_totals(contract.split)


def _read_pinned_root_file(path: Path) -> bytes:
    """Read a local ChemKED file with the same cap and TOCTOU checks as fetches."""
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot read local ChemKED file {path}: {exc}") from exc
    if before.st_size > _MAX_FILE_BYTES:
        raise ValueError(f"local ChemKED file {path} is {before.st_size} bytes, over the {_MAX_FILE_BYTES} cap")
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_FILE_BYTES + 1)
        after = path.stat()
    except OSError as exc:
        raise ValueError(f"cannot read local ChemKED file {path}: {exc}") from exc
    if len(data) > _MAX_FILE_BYTES or after.st_size > _MAX_FILE_BYTES:
        size = max(len(data), after.st_size)
        raise ValueError(f"local ChemKED file {path} is {size} bytes, over the {_MAX_FILE_BYTES} cap")
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or after.st_size != len(data):
        raise ValueError(f"local ChemKED file {path} changed while reading")
    return data


def write_schema(path: Path) -> None:
    """Publish the exact JSON Schema used by Python validation."""
    path.write_text(json.dumps(Contract.model_json_schema(), indent=2) + "\n")
