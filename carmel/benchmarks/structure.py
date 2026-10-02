"""Structure-only rules for the frozen n-heptane benchmark (requires agents extra)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType

from rdkit import Chem


@dataclass(frozen=True)
class Features:
    atoms: Mapping[str, int]
    hydroperoxides: int
    hydroperoxide_on_carbonyl: bool
    peroxy_radicals: int
    carbon_radicals: int
    radicals: int
    carbonyls: int
    alkenes: int
    ether_rings: int
    acids: int
    linear_carbon: bool


def molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"UNRESOLVED: invalid or disconnected SMILES: {smiles}")
    return mol


@cache
def canonical(smiles: str) -> str:
    return str(Chem.MolToSmiles(molecule(smiles), isomericSmiles=False))


@cache
def features(smiles: str) -> Features:
    mol = molecule(smiles)
    atoms = Counter(a.GetSymbol() for a in mol.GetAtoms())
    atoms["H"] += sum(a.GetTotalNumHs() for a in mol.GetAtoms())

    def matches(pattern: str) -> int:
        return len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))

    carbonyl_carbons = {match[0] for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[C]=[O]"))}
    hydroperoxide_carbons = {match[0] for match in mol.GetSubstructMatches(Chem.MolFromSmarts("[C]-[O]-[O;H1]"))}

    carbon_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    carbon_bonds = [b for b in mol.GetBonds() if b.GetBeginAtom().GetSymbol() == b.GetEndAtom().GetSymbol() == "C"]
    linear = bool(carbon_atoms) and len(carbon_bonds) == len(carbon_atoms) - 1
    linear = linear and all(sum(n.GetSymbol() == "C" for n in a.GetNeighbors()) <= 2 for a in carbon_atoms)
    return Features(
        atoms=MappingProxyType(dict(atoms)),
        hydroperoxides=matches("[C]-[O]-[O;H1]"),
        hydroperoxide_on_carbonyl=bool(carbonyl_carbons & hydroperoxide_carbons),
        peroxy_radicals=matches("[C]-[O]-[O;X1;H0]"),
        carbon_radicals=sum(a.GetNumRadicalElectrons() for a in carbon_atoms),
        radicals=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
        carbonyls=matches("[C]=[O]"),
        alkenes=matches("[C]=[C]"),
        ether_rings=matches("[C;R]-[O;R]-[C;R]"),
        acids=matches("[C](=[O])-[O;H1]"),
        linear_carbon=linear,
    )


def reaction_signature(reactants: tuple[str, ...], products: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Ignore labels, rates, direction, and unassigned stereochemistry; retain multiplicity."""
    return tuple(sorted((tuple(sorted(map(canonical, reactants))), tuple(sorted(map(canonical, products))))))


def classify(reactants: tuple[str, ...], products: tuple[str, ...]) -> frozenset[str]:
    """Recognize net structural classes in either orientation; require elemental balance."""
    left, right = [features(s) for s in reactants], [features(s) for s in products]
    if sum((Counter(f.atoms) for f in left), Counter()) != sum((Counter(f.atoms) for f in right), Counter()):
        return frozenset()
    found: set[str] = set()
    for source, target, target_smiles in ((left, right, products), (right, left, reactants)):
        if len(source) != 1:
            continue
        src = source[0]
        if src.atoms["C"] != 7 or not src.linear_carbon:
            continue
        pooh = (
            src.atoms == Counter({"C": 7, "H": 15, "O": 4})
            and src.hydroperoxides == 2
            and not src.radical_on_hydroperoxy_carbon
            and src.carbon_radicals == src.radicals == 1
            and src.carbonyls == src.alkenes == src.ether_rings == 0
        )
        ooqooh = (
            src.atoms == Counter({"C": 7, "H": 15, "O": 4})
            and src.hydroperoxides == src.peroxy_radicals == src.radicals == 1
            and src.carbon_radicals == src.carbonyls == src.alkenes == src.ether_rings == 0
        )
        carbon_products = [f for f in target if f.atoms.get("C", 0)]
        small = sorted(canonical(s) for s, f in zip(target_smiles, target, strict=True) if not f.atoms.get("C", 0))
        if len(carbon_products) == 1:
            dst = carbon_products[0]
            if dst.atoms["C"] != 7 or not dst.linear_carbon:
                continue
            if ooqooh and len(target) == 1 and dst.hydroperoxides == 2 and dst.carbon_radicals == dst.radicals == 1:
                found.add("G1")
            if pooh and small == ["[OH]"] and dst.ether_rings == dst.hydroperoxides == 1 and dst.radicals == 0:
                found.add("G2")
            if small == ["[O]O"] and dst.hydroperoxides == dst.alkenes == 1 and dst.radicals == 0:
                if pooh:
                    found.add("G3")
                if ooqooh:
                    found.add("G4")
        elif len(carbon_products) >= 2 and pooh:
            found.add("G5")
        elif len(carbon_products) == 2 and len(target) == 2 and src.atoms == Counter({"C": 7, "H": 14, "O": 3}):
            acids = [f for f in target if f.acids == f.carbonyls == 1]
            non_acid_carbonyls = [f for f in target if f.acids == 0 and f.carbonyls == 1]
            if (
                src.hydroperoxides == src.carbonyls == 1
                and src.radicals == 0
                and not src.hydroperoxide_on_carbonyl
                and len(acids) == len(non_acid_carbonyls) == 1
                and all(f.radicals == 0 for f in target)
            ):
                found.add("G6")
    return frozenset(found)


def ketohydroperoxide(label: str) -> str | None:
    """LLNL N prefix is a straight chain; KETij places C=O at i and OOH at j."""
    match = re.fullmatch(r"N?C([3-7])KET([1-7])([1-7])", label)
    if match is None:
        return None
    length, carbonyl, hydroperoxy = map(int, match.groups())
    if carbonyl == hydroperoxy or max(carbonyl, hydroperoxy) > length:
        return None
    chain = ["C" for _ in range(length)]
    chain[carbonyl - 1] += "(=O)"
    chain[hydroperoxy - 1] += "(OO)"
    return canonical("".join(chain))
