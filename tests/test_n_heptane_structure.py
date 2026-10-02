"""Tests exercise structural classes using author-sourced SMILES and tiny excerpts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from carmel.benchmarks.structure import canonical, classify, features, ketohydroperoxide, molecule, reaction_signature

ROOT = Path(__file__).resolve().parents[1]
RAW = json.loads((ROOT / "benchmarks/n-heptane-low-t/contract.json").read_text())
SMILES = {s["label"]: s["smiles"] for s in RAW["new_species"]}


@pytest.mark.parametrize("gt", RAW["ground_truth"], ids=lambda g: g["id"])
def test_class_rules_accept_every_member(gt: dict) -> None:
    for member in gt["members"]:
        reactants = tuple(SMILES[s] for s in member["reactants"])
        products = tuple(SMILES[s] for s in member["products"])
        assert classify(reactants, products) == {gt["id"]}
        assert classify(products, reactants) == {gt["id"]}


def test_tiny_chemkin_excerpts_match_committed_members() -> None:
    fixture = ROOT / "tests/fixtures/n_heptane/new-reactions.txt"
    equations = {m["equation"] for g in RAW["ground_truth"] for m in g["members"]}
    for line in fixture.read_text().splitlines():
        if line and not line.startswith("!"):
            assert line.split()[0] in equations


@pytest.mark.parametrize(
    "reactants,products",
    [
        # Ordinary QOOH cyclic ether (only one OOH).
        (("C(OO)C[CH]CCCC",), ("C1CC(CCCC)O1", "[OH]")),
        # Ordinary RO2 <=> QOOH H shift.
        (("CCCCCCCO[O]",), ("C(OO)C[CH]CCCC",)),
        # Conventional OOQOOH => KHP + OH.
        (("C(OO)CC(O[O])CCCC",), ("C(=O)CC(OO)CCCC", "[OH]")),
        # Different carbon skeleton size must not expand the frozen case.
        (("C(OO)CC(O[O])CCC",), ("C(OO)CC(OO)[CH]CC",)),
        # An unbalanced proposal earns no structural credit.
        (("C(OO)CC(O[O])CCCC",), ("C(OO)CC(OO)[CH]CCC", "[OH]")),
        # Two source molecules are not these net single-source classes.
        (("CCCCCCC", "O"), ("CCCCCCCO",)),
        # Branched C7 OOQOOH cannot be counted as straight-chain n-heptane.
        (("CC(C)(OO)CC(O[O])CC",), ("CC(C)(OO)[CH]C(OO)CC",)),
    ],
)
def test_reject_conventional_chain_and_out_of_scope_reactions(reactants: tuple, products: tuple) -> None:
    assert classify(reactants, products) == set()


@pytest.mark.parametrize("label", ["garbage", "C7KET11", "C3KET17"])
def test_grammar_refuses_unknown_or_impossible_labels(label: str) -> None:
    assert ketohydroperoxide(label) is None


def test_llnl_ket_grammar_matches_all_new_glossary_isomers() -> None:
    for species in RAW["old_species"]:
        label = species["label"]
        if re.fullmatch(r"NC7KET\d\d", label):
            assert ketohydroperoxide(label) == species["smiles"]
    assert ketohydroperoxide("NC7KET13") == canonical(SMILES["C7KET13"])
    assert ketohydroperoxide("C7KET13") == canonical(SMILES["C7KET13"])


@pytest.mark.parametrize("smiles", ["invalid", "C.C"])
def test_bad_structure_is_explicitly_unresolved(smiles: str) -> None:
    with pytest.raises(ValueError, match="UNRESOLVED"):
        molecule(smiles)


def test_signature_ignores_labels_orientation_and_order_but_keeps_multiplicity() -> None:
    assert reaction_signature(("CCO",), ("C", "O")) == reaction_signature(("O", "C"), ("OCC",))
    assert reaction_signature(("CCO",), ("C", "O")) != reaction_signature(("CCO",), ("C", "O", "O"))
    assert features("[CH2]CC(OO)CC(OO)CC").carbon_radicals == 1


def test_cached_atom_counts_are_immutable() -> None:
    atoms = features("CC").atoms

    with pytest.raises(TypeError):
        atoms["C"] = 99


@pytest.mark.parametrize(
    "label,expected",
    [
        ("NC7KET12", "C(=O)C(OO)CCCCC"),
        ("NC7KET31", "C(OO)CC(=O)CCCC"),
        ("NC7KET41", "C(OO)CCC(=O)CCC"),
    ],
)
def test_ket_grammar_carbonyl_and_hydroperoxide_positions(label: str, expected: str) -> None:
    assert ketohydroperoxide(label) == canonical(expected)


def test_conventional_alpha_hydroperoxy_intermediate_gets_no_alternative_credit() -> None:
    # Bugler2015 section2.2.2/Fig12: the conventional two-step KHP route.
    ooqooh = ("C(OO)CC(O[O])CCCC",)
    alpha = ("[CH](OO)CC(OO)CCCC",)
    assert features(alpha[0]).radical_on_hydroperoxy_carbon
    assert classify(ooqooh, alpha) == set()
    assert classify(alpha, ooqooh) == set()
    assert not features(SMILES["C7H13Q13-5"]).radical_on_hydroperoxy_carbon


@pytest.mark.parametrize(
    "reactants,products",
    [
        # An extra sulfur preserves every legacy scalar check but is outside C7H15O4.
        (("C(OO)CC(O[O])CCCC[SH]",), ("C(OO)CC(OO)[CH]CCC[SH]",)),
        (("[SH]CCCC[CH]C(COO)OO",), ("[SH]CCCCC1OCC1OO", "[OH]")),
        (("[SH]CCCC[CH]C(COO)OO",), ("[SH]CCCCC=CCOO", "[O]O")),
        (("[SH]CCCCCC(COO)O[O]",), ("[SH]CCCCC=CCOO", "[O]O")),
        (("[SH]CCCC[CH]C(COO)OO",), ("[SH]CCCC[CH]C=O", "[CH2]OO", "[OH]")),
    ],
)
def test_heteroatom_carrying_lookalikes_are_not_benchmark_members(
    reactants: tuple[str, ...], products: tuple[str, ...]
) -> None:
    assert classify(reactants, products) == set()


@pytest.mark.parametrize(
    "reactants,products",
    [
        # Wrong source formula: C7H14O3S, not frozen C7H14O3.
        (("C(=O)CC(OO)CCCC[SH]",), ("[SH]CC(=O)O", "CCCCC=O")),
        # Both product carbonyls sit on glyoxylic acid; the alkane is not a carbonyl product.
        (("C(=O)CC(OO)CCCC",), ("O=CC(=O)O", "CCCCC")),
        # An acyl hydroperoxide is not a Korcek hydroperoxide source.
        (("CCCCCCC(=O)OO",), ("CC(=O)O", "CCCCC=O")),
    ],
)
def test_g6_requires_frozen_source_and_split_carbonyl_products(
    reactants: tuple[str, ...], products: tuple[str, ...]
) -> None:
    assert classify(reactants, products) == set()
