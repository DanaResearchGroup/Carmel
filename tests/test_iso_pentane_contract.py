"""Network-free case-3 contract and shared-rule regressions."""

import hashlib
import io
import json
import runpy
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from carmel.benchmarks import iso_pentane
from carmel.benchmarks.iso_pentane import Contract, load_contract, xml_temperature_max
from carmel.benchmarks.n_pentane import require_structural_absence, split_totals
from carmel.benchmarks.schema import benchmark_contract_schema, validate_contract_json

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/iso-pentane-low-t/contract.json"


def test_contract_validates_and_preserves_case2_pins_and_scoring():
    raw = json.loads(PATH.read_bytes())
    jsonschema.validate(raw, benchmark_contract_schema())
    contract = validate_contract_json(PATH.read_bytes())
    assert isinstance(contract, Contract)
    assert split_totals(contract.split) == Contract.expected_totals
    case2 = json.loads((ROOT / "benchmarks/n-pentane-low-t/contract.json").read_bytes())
    # Case 3 carries case 2's scoring as frozen before revision 2 added target chemistry.
    assert raw["scoring"] == {k: v for k, v in case2["scoring"].items() if k != "target_chemistry"}
    assert "revision" not in raw
    assert raw["mechanisms"]["v_new"] == case2["mechanisms"]["v_new"]
    assert raw["split"]["archive"] == case2["split"]["archive"]
    assert contract.conversion["v_old"].converted_reactions == 3796 + 3068
    assert contract.thermo_sources["v_old"].covered_mechanism_species == 874


def test_contract_load_is_independent_of_rdkit():
    code = f"""import sys
sys.modules["rdkit"] = None
from pathlib import Path
from carmel.benchmarks.schema import validate_contract_json
assert validate_contract_json(Path({str(PATH)!r}).read_bytes()).case == "iso-pentane-low-t"
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


@pytest.mark.parametrize("group,count", [("G1", 33), ("G2", 33), ("G3", 14), ("G4", 14)])
def test_all_members_match_both_orientations_and_reject_linear_scope(group, count):
    pytest.importorskip("rdkit")
    from carmel.benchmarks.structure import classify

    contract = load_contract(PATH)
    smiles = {s.label: s.smiles for s in contract.new_species}
    truth = next(t for t in contract.ground_truth if t.id == group)
    assert truth.member_count == truth.resolved_count == count
    assert len(truth.members) == count
    for member in truth.members:
        left = tuple(smiles[s] for s in member.reactants)
        right = tuple(smiles[s] for s in member.products)
        assert classify(left, right, carbon_count=5, skeleton="iso-pentane") == {group}
        assert classify(right, left, carbon_count=5, skeleton="iso-pentane") == {group}
        assert not classify(left, right, carbon_count=5)


@pytest.mark.parametrize(
    "left,right",
    [
        (("CCC(COO)CO[O]",), ("CCC(C=O)COO", "[OH]")),  # conventional KHP
        (("CCC(C)CO[O]",), ("CCC([CH2])COO",)),  # RO2 to QOOH
        (("CCC([CH2])COO",), ("CCC1COC1", "[OH]")),  # ordinary cyclic ether
        (("CC(C)(COO)CO[O]",), ("[CH2]C(C)(COO)COO",)),  # neo-C5
        (("C(OO)CC(O[O])CC",), ("C(OO)CC(OO)[CH]C",)),  # n-C5
        (("O=O",), ("[O]", "[O]")),
    ],
)
def test_conventional_and_wrong_skeleton_negatives(left, right):
    pytest.importorskip("rdkit")
    from carmel.benchmarks.structure import classify

    assert not classify(left, right, carbon_count=5, skeleton="iso-pentane")


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_old_mapping_proves_mismatch_but_never_guesses_unknown(reverse):
    pytest.importorskip("rdkit")
    from carmel.benchmarks.structure import reaction_signature

    candidate = dict(reactants=("A",), products=("B",), equation="A=B", line=1)
    prior = dict(reactants=("X",), products=("UNKNOWN",), equation="X=UNKNOWN", line=2)
    if reverse:
        prior["reactants"], prior["products"] = prior["products"], prior["reactants"]
    forms = {x: {"C": 5, "H": 11, "O": 4} for x in ("A", "B", "X", "UNKNOWN")}
    signature = reaction_signature(("CCC(COO)CO[O]",), ("CC[C](COO)COO",))
    # A neo skeleton cannot be either side of this iC5 reaction.
    require_structural_absence(candidate, forms, signature, [prior], forms, {"X": "CC(C)(COO)CO[O]"})
    with pytest.raises(ValueError, match="UNRESOLVED old"):
        require_structural_absence(candidate, forms, signature, [prior], forms, {"X": "CCC(COO)CO[O]"})


@pytest.mark.parametrize(
    "change,message",
    [
        ("bound", "below reviewed Tmid"),
        ("missing_bound", "every holdout file"),
        ("max", "disagrees"),
        ("review", "retain every reviewed"),
        ("thermo", "both thermo"),
        ("coverage", "cover all"),
        ("source_count", "source reaction"),
        ("converted_count", "explicit reverse"),
        ("normalization", "both inputs"),
        ("hash", "raw hash"),
        ("registry", "retain every reviewed"),
        ("warning", "unreviewed NASA validation"),
    ],
)
def test_conversion_and_temperature_invariants_fail_closed(change, message):
    raw = json.loads(PATH.read_bytes())
    audit = raw["conversion"]["v_old"]
    if change == "bound":
        raw["holdout_max_temperature_k"] = 1388
        raw["holdout_file_max_temperature_k"]["x00011009.xml"] = 1388
    elif change == "missing_bound":
        raw["holdout_file_max_temperature_k"].pop("x00011009.xml")
    elif change == "max":
        raw["holdout_max_temperature_k"] = 1099
    elif change == "review":
        audit["reviewed_warnings"] = []
    elif change == "registry":
        audit["reviewed_warnings"][0]["requires_below_tmid"] = False
    elif change == "warning":
        audit["warnings"][0] = "unexpected warning"
    elif change == "thermo":
        raw["thermo_sources"].pop("v_old")
    elif change == "coverage":
        raw["thermo_sources"]["v_old"]["covered_mechanism_species"] -= 1
    elif change == "source_count":
        audit["source_reaction_entries"] += 1
    elif change == "converted_count":
        audit["converted_reactions"] += 1
    elif change == "normalization":
        audit["normalization"].pop("thermo")
    else:
        audit["normalization"]["thermo"]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        Contract.model_validate(raw)


def _xml(value="1100", units="K", axis='id="x1"'):
    return (
        f'<experiment><dataGroup><property name="composition" id="x2"/>'
        f'<property name="temperature" {axis} units="{units}"/>'
        f"<dataPoint><x1>{value}</x1></dataPoint></dataGroup></experiment>"
    ).encode()


@pytest.mark.parametrize("raw", [_xml("nan"), _xml("0"), _xml(units="C"), _xml(axis=""), b"<experiment/>"])
def test_xml_temperature_bounds_refuse_incomplete_or_invalid_data(raw):
    with pytest.raises(ValueError):
        xml_temperature_max(raw)


@pytest.mark.parametrize("drift", [False, True])
def test_split_loader_verifies_actual_temperature_bound(tmp_path, monkeypatch, drift):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x.xml", _xml())
    raw = archive.getvalue()
    path = tmp_path / "archive.zip"
    path.write_bytes(raw)
    contract = SimpleNamespace(holdout_file_max_temperature_k={"x.xml": 1000 if drift else 1100})

    def verify(c, data):
        assert c is contract and hashlib.sha256(data).digest() == hashlib.sha256(raw).digest()
        path.unlink()  # Temperature replay must consume the same verified snapshot.
        return {"holdout_rows": 1}

    monkeypatch.setattr(iso_pentane, "load_c5_split", verify)
    if drift:
        with pytest.raises(ValueError, match="temperature maximum changed"):
            iso_pentane.load_split(contract, path)
    else:
        assert iso_pentane.load_split(contract, path) == {"holdout_rows": 1}


def test_shared_thermo_formula_parser_accepts_integer_decimal_counts_only():
    parser = runpy.run_path(str(ROOT / "benchmarks/n-heptane-low-t/reproduce.py"))["thermo"]
    header = b"C6H5CH2           IU3/03C  7.H  7.   0.   0.G   250.000  6000.000 1000.        1"
    assert parser(header) == {"C6H5CH2": {"C": 7, "H": 7}}
    with pytest.raises(ValueError, match="noninteger"):
        parser(header.replace(b" 7.", b"7.5", 1))


def _member_fixture():
    """Four frozen members (one per group), as synthetic network-free mechanism inputs."""
    from carmel.benchmarks.structure import features

    contract = json.loads(PATH.read_bytes())
    smiles = {s["label"]: s["smiles"] for s in contract["new_species"]}
    reactions, groups = [], {}
    for truth in contract["ground_truth"]:
        member = truth["members"][0]
        reaction = {k: member[k] for k in ("equation", "line", "reactants", "products")}
        if reaction["equation"] not in groups:
            reactions.append(reaction)
        groups.setdefault(reaction["equation"], set())
    for truth in contract["ground_truth"]:
        for member in truth["members"]:
            if member["equation"] in groups:
                groups[member["equation"]].add(truth["id"])
    labels = sorted({n for r in reactions for n in (*r["reactants"], *r["products"])})
    formulas = {label: dict(Counter(features(smiles[label]).atoms)) for label in labels}
    glossary = {label: {"smiles": smiles[label]} for label in labels}
    return reactions, groups, labels, formulas, glossary


def _patched_mapped_truth(monkeypatch):
    namespace = runpy.run_path(str(ROOT / "benchmarks/iso-pentane-low-t/reproduce.py"))
    mapped_truth = namespace["mapped_truth"]
    module_globals = mapped_truth.__globals__
    monkeypatch.setitem(module_globals, "mechanism", lambda raw: raw)
    monkeypatch.setitem(module_globals, "thermo_inventory", lambda raw: (raw, None))
    heptane = dict(module_globals["HEPTANE"])
    heptane["glossary"] = lambda glossary: glossary
    heptane["inventory"] = lambda labels, dictionary, _formulas: [
        {"label": label, "smiles": dictionary.get(label, {}).get("smiles")} for label in labels
    ]
    monkeypatch.setitem(module_globals, "HEPTANE", heptane)
    return mapped_truth


def test_member_selection_finds_every_group_in_either_written_orientation(monkeypatch):
    pytest.importorskip("rdkit")
    reactions, groups, labels, formulas, glossary = _member_fixture()
    mapped_truth = _patched_mapped_truth(monkeypatch)
    # Write one member backwards: selection must not depend on the source being on the left.
    flipped = [dict(r) for r in reactions]
    flipped[1]["reactants"], flipped[1]["products"] = reactions[1]["products"], reactions[1]["reactants"]
    for written in (reactions, flipped):
        truth = mapped_truth(([], []), (labels, written), {}, formulas, glossary)["ground_truth"]
        selected = {t["id"]: {m["equation"] for m in t["members"]} for t in truth}
        expected = {g: {eq for eq, gs in groups.items() if g in gs} for g in ("G1", "G2", "G3", "G4")}
        assert selected == expected


def test_member_selection_refuses_an_unresolved_c5_source(monkeypatch):
    pytest.importorskip("rdkit")
    reactions, _groups, labels, formulas, glossary = _member_fixture()
    mapped_truth = _patched_mapped_truth(monkeypatch)
    source = next(label for label in labels if formulas[label] == {"C": 5, "H": 11, "O": 4})
    unresolved = {label: entry for label, entry in glossary.items() if label != source}
    with pytest.raises(ValueError, match="UNRESOLVED C5 source"):
        mapped_truth(([], []), (labels, reactions), {}, formulas, unresolved)


def test_member_selection_refuses_a_reaction_already_in_v_old(monkeypatch):
    pytest.importorskip("rdkit")
    reactions, _groups, labels, formulas, glossary = _member_fixture()
    mapped_truth = _patched_mapped_truth(monkeypatch)
    prior = reactions[0]
    old_labels = sorted({*prior["reactants"], *prior["products"]})
    with pytest.raises(ValueError, match="baseline reaction signature already present"):
        mapped_truth((old_labels, [prior]), (labels, reactions), formulas, formulas, glossary)


def test_case3_contract_is_digest_frozen_and_rejects_an_unrecorded_edit():
    raw = json.loads(PATH.read_bytes())
    assert validate_contract_json(json.dumps(raw)).case == "iso-pentane-low-t"
    raw["leakage_limits"] = [*raw["leakage_limits"], "unrecorded"]
    with pytest.raises(ValueError, match="UNRECORDED CONTRACT CHANGE"):
        validate_contract_json(json.dumps(raw))
