"""Shared immutable primitives and invariants for frozen benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, NoReturn, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class Identity(FrozenModel):
    status: Literal["UNVERIFIED"]
    historical_identity_caveat: str
    attempts: tuple[str, ...]
    evidence: tuple[str, ...]


def validate_mechanism_pair(mechanisms: Mapping[Any, object]) -> None:
    if set(mechanisms) != {"v_old", "v_new"}:
        raise ValueError("both mechanism pins are required")


def validate_species_inventories(
    new_species: tuple[Species, ...], old_species: tuple[Species, ...], *, expected_old_species: int
) -> dict[str, Species]:
    new = {species.label: species for species in new_species}
    if len(new) != len(new_species) or len({species.label for species in old_species}) != len(old_species):
        raise ValueError("species labels must be unique within each mechanism")
    if len(old_species) != expected_old_species:
        raise ValueError("absence audit must inventory every old species")
    return new


def validate_ground_truth_counts(
    members: tuple[Member, ...],
    frozen_counts: tuple[int, int, int, int],
    new_species: Mapping[str, Species],
) -> set[str]:
    labels = {label for member in members for label in (*member.reactants, *member.products)}
    if labels - new_species.keys():
        raise ValueError("member references an uninventoried species")
    resolved = sum(
        all(new_species[label].status == "RESOLVED" for label in (*member.reactants, *member.products))
        for member in members
    )
    resolved_species = sum(new_species[label].status == "RESOLVED" for label in labels)
    if frozen_counts != (len(members), resolved, len(labels), resolved_species):
        raise ValueError("ground-truth counts do not reproduce")
    return labels
