"""Network-free checks for the frozen n-pentane benchmark contract."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

pytest.importorskip("rdkit")

from carmel.benchmarks import n_pentane
from carmel.benchmarks.n_pentane import Contract, SplitFile, load_contract, split_totals
from carmel.benchmarks.structure import classify
from carmel.services.respecth import RespecthRefusal, RespecthRefusalReason

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/n-pentane-low-t/contract.json"


@pytest.fixture(scope="module")
def contract() -> Contract:
    return load_contract(CONTRACT_PATH)


def test_contract_validates_and_reproduces_frozen_totals(contract: Contract) -> None:
    raw = json.loads(CONTRACT_PATH.read_text())
    schema = Contract.model_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(raw, schema)
    assert split_totals(contract.split) == {
        "development_files": 26,
        "development_rows": 226,
        "development_observations": 226,
        "development_supported_files": 0,
        "holdout_files": 6,
        "holdout_rows": 144,
        "holdout_observations": 2080,
        "holdout_supported_files": 6,
        "holdout_low_temperature_rows": 72,
        "holdout_low_temperature_observations": 1040,
    }
    assert [item.member_count for item in contract.ground_truth] == [30, 30, 12, 12]
    assert contract.scoring.requires == "jsr-adapter"
    assert contract.identity.status == "UNVERIFIED"


def test_contract_rejects_a_doi_reused_across_study_groups() -> None:
    raw = json.loads(CONTRACT_PATH.read_text())
    raw["split"]["studies"][1]["doi"] = raw["split"]["studies"][0]["doi"]

    with pytest.raises(ValueError, match="source DOI must not cross study groups"):
        Contract.model_validate(raw)


def test_all_members_are_resolved_and_accepted_by_their_shared_class_rule(contract: Contract) -> None:
    species = {item.label: item.smiles for item in contract.new_species}
    for truth in contract.ground_truth:
        assert truth.resolved_count == truth.member_count
        assert truth.resolved_species_count == truth.member_species_count
        for member in truth.members:
            reactants = tuple(species[label] for label in member.reactants)
            products = tuple(species[label] for label in member.products)
            assert all(reactants) and all(products)
            assert classify(reactants, products, carbon_count=5) == {truth.id}
            assert classify(products, reactants, carbon_count=5) == {truth.id}


def test_tiny_excerpts_are_frozen_members_or_a_conventional_negative(contract: Contract) -> None:
    equations = {member.equation for truth in contract.ground_truth for member in truth.members}
    lines = (ROOT / "tests/fixtures/n_pentane/reaction-excerpts.txt").read_text().splitlines()
    excerpt_equations = [line.split()[0] for line in lines if line and not line.startswith("!")]
    assert excerpt_equations[:4] == [
        "C5H10OOH1-2O2<=>C5H93-1,2OOH",
        "C5H91-2,3OOH<=>C5H9O1-2OOH-3+OH",
        "C5H91-2,3OOH<=>C5H91-3OOH+HO2",
        "C5H10OOH1-2O2<=>C5H92-1OOH+HO2",
    ]
    assert set(excerpt_equations[:4]) <= equations
    assert excerpt_equations[4] not in equations


@pytest.mark.parametrize(
    "reactants,products",
    [
        (("CCCC(COO)O[O]",), ("CCCC(C=O)OO", "[OH]")),
        (("C(OO)C[CH]CC",), ("C1CC(CC)O1", "[OH]")),
        (("CCC(COO)O[O]",), ("C[CH]C(OO)COO",)),
        (("CC(C)(OO)C(O[O])C",), ("CC(C)(OO)[CH]C(OO)",)),
    ],
)
def test_shared_rules_reject_conventional_and_out_of_scope_c5_chains(
    reactants: tuple[str, ...], products: tuple[str, ...]
) -> None:
    assert classify(reactants, products, carbon_count=5) == set()


def test_data_acceptance_freezes_every_file_and_typed_refusal(contract: Contract) -> None:
    summary = Counter(
        (study.set, item.acceptance, item.refusal_reason) for study in contract.split.studies for item in study.files
    )
    assert summary == Counter(
        {
            ("development", "refused", "unmapped_property"): 9,
            ("development", "refused", "rcm_thermo_unavailable"): 11,
            ("development", "refused", "history_nonmonotone_time"): 3,
            ("development", "refused", "unmapped_ignition_definition"): 3,
            ("holdout", "supported", None): 6,
        }
    )


def test_pin_verification_fails_loudly(contract: Contract) -> None:
    pin = contract.mechanisms["v_old"]
    with pytest.raises(ValueError, match="PIN CHANGED"):
        pin.verify(b"changed mechanism bytes")


def _record(*temperatures: int, observations: int = 2) -> SimpleNamespace:
    points = tuple(
        SimpleNamespace(
            coordinates=(
                SimpleNamespace(
                    axis_id="temperature",
                    value=SimpleNamespace(canonical_decimal_value=str(temperature)),
                ),
            ),
            observations=(object(),) * observations,
        )
        for temperature in temperatures
    )
    series = SimpleNamespace(constants=(), points=points)
    return SimpleNamespace(envelope=SimpleNamespace(series=(series,)))


def _archive_contract(tmp_path: Path, *, acceptance: str = "supported", reason: str | None = None):
    member = b'<experiment><dataGroup><property name="composition"/><dataPoint/><dataPoint/></dataGroup></experiment>'
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as target:
        target.writestr("xunit.xml", member)
    archive_bytes = archive_path.read_bytes()
    archive = SimpleNamespace(
        filename=archive_path.name,
        osf_path="/fixture.zip",
        osf_file_id="0" * 24,
        osf_version=1,
        bytes=len(archive_bytes),
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        verify=lambda data: None,
    )
    item = SimpleNamespace(
        path="xunit.xml",
        sha256=hashlib.sha256(member).hexdigest(),
        rows=2,
        observations=4,
        low_temperature_rows=1,
        low_temperature_observations=2,
        acceptance=acceptance,
        refusal_reason=reason,
    )
    study = SimpleNamespace(set="holdout", files=(item,))
    contract = SimpleNamespace(split=SimpleNamespace(archive=archive, studies=(study,)))
    return contract, archive_path, item, study


def test_split_loader_reverifies_and_replays_supported_member(tmp_path: Path, monkeypatch) -> None:
    contract, archive_path, _, _ = _archive_contract(tmp_path)
    monkeypatch.setattr(n_pentane, "parse_series_record", lambda *args: _record(700, 1000))
    assert n_pentane.load_split(contract, archive_path) == {
        "development_supported_files": 0,
        "holdout_supported_files": 1,
        "holdout_files": 1,
        "holdout_rows": 2,
        "holdout_observations": 4,
        "holdout_low_temperature_rows": 1,
        "holdout_low_temperature_observations": 2,
    }


def test_split_loader_uses_the_verified_archive_bytes(tmp_path: Path, monkeypatch) -> None:
    contract, archive_path, _, _ = _archive_contract(tmp_path)
    archive_bytes = archive_path.read_bytes()
    archive_path.unlink()
    monkeypatch.setattr(n_pentane, "parse_series_record", lambda *args: _record(700, 1000))

    assert n_pentane.load_split(contract, archive_bytes)["holdout_supported_files"] == 1


def test_split_loader_replays_typed_refusal(tmp_path: Path, monkeypatch) -> None:
    contract, archive_path, item, study = _archive_contract(tmp_path, acceptance="refused", reason="unmapped_property")
    study.set = "development"
    item.observations = 2
    item.low_temperature_rows = item.low_temperature_observations = 0

    def refuse(*args):
        raise RespecthRefusal(RespecthRefusalReason.UNMAPPED_PROPERTY, "fixture")

    monkeypatch.setattr(n_pentane, "parse_idt_record", refuse)
    totals = n_pentane.load_split(contract, archive_path)
    assert totals["development_files"] == 1
    assert totals["development_supported_files"] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("hash", "SPLIT BLOB CHANGED"),
        ("rows", "row count changed"),
        ("acceptance", "DATA ACCEPTANCE CHANGED"),
        ("counts", "parsed counts changed"),
    ],
)
def test_split_loader_fails_loudly_on_drift(tmp_path: Path, monkeypatch, change: str, message: str) -> None:
    contract, archive_path, item, _ = _archive_contract(tmp_path)
    monkeypatch.setattr(n_pentane, "parse_series_record", lambda *args: _record(700, 1000))
    if change == "hash":
        item.sha256 = "f" * 64
    elif change == "rows":
        item.rows = 3
    elif change == "acceptance":
        item.acceptance = "refused"
        item.refusal_reason = "unmapped_property"
    else:
        item.observations = 5
    with pytest.raises(ValueError, match=message):
        n_pentane.load_split(contract, archive_path)


def test_schema_writer_emits_the_contract_schema(tmp_path: Path) -> None:
    target = tmp_path / "schema.json"
    n_pentane.write_schema(target)
    assert json.loads(target.read_text())["title"] == "Contract"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("mechanism", "both mechanism pins"),
        ("support", "sole support source"),
        ("truths", "exactly the four"),
        ("primary", "all and only"),
        ("species", "species labels must be unique"),
        ("formula", "reactant formula"),
        ("label", "uninventoried species"),
        ("counts", "ground-truth counts"),
        ("resolution", "fewer than 70%"),
        ("path", "whole studies must be disjoint"),
        ("study", "whole studies must be disjoint"),
        ("refused", "more than one third"),
        ("totals", "split counts do not reproduce"),
    ],
)
def test_contract_invariants_fail_closed(change: str, message: str) -> None:
    raw = json.loads(CONTRACT_PATH.read_text())
    if change == "mechanism":
        raw["mechanisms"].pop("v_old")
    elif change == "support":
        raw["support_sources"] = []
    elif change == "truths":
        raw["ground_truth"].pop()
    elif change == "primary":
        raw["scoring"]["primary_types"].pop()
    elif change == "species":
        raw["new_species"].append(raw["new_species"][0])
    elif change == "formula":
        raw["ground_truth"][0]["reactant_formula"]["C"] = 4
    elif change == "label":
        raw["ground_truth"][0]["members"][0]["reactants"][0] = "UNKNOWN"
    elif change == "counts":
        raw["ground_truth"][0]["member_count"] += 1
    elif change == "resolution":
        for species in raw["new_species"]:
            species["status"] = "UNRESOLVED"
            species["smiles"] = None
        for truth in raw["ground_truth"]:
            truth["resolved_count"] = truth["resolved_species_count"] = 0
    elif change == "path":
        raw["split"]["studies"][1]["files"][0]["path"] = raw["split"]["studies"][0]["files"][0]["path"]
    elif change == "study":
        raw["split"]["studies"][1]["id"] = raw["split"]["studies"][0]["id"]
    elif change == "refused":
        for item in raw["split"]["studies"][1]["files"][:3]:
            item["acceptance"] = "refused"
            item["refusal_reason"] = "unmapped_property"
    else:
        raw["split"]["studies"][0]["files"][0]["rows"] += 1
    with pytest.raises(ValueError, match=message):
        Contract.model_validate(raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("directory", "plain XML"),
        ("suffix", "plain XML"),
        ("reason", "refused files require"),
        ("supported_reason", "supported files forbid"),
        ("low_rows", "cannot exceed"),
        ("low_observations", "cannot exceed"),
    ],
)
def test_split_file_invariants_fail_closed(change: str, message: str) -> None:
    raw = {
        "path": "x.xml",
        "sha256": "0" * 64,
        "rows": 2,
        "observations": 4,
        "low_temperature_rows": 1,
        "low_temperature_observations": 2,
        "acceptance": "supported",
        "refusal_reason": None,
    }
    if change == "directory":
        raw["path"] = "nested/x.xml"
    elif change == "suffix":
        raw["path"] = "x.txt"
    elif change == "reason":
        raw["acceptance"] = "refused"
    elif change == "supported_reason":
        raw["refusal_reason"] = "unmapped_property"
    elif change == "low_rows":
        raw["low_temperature_rows"] = 3
    else:
        raw["low_temperature_observations"] = 5
    with pytest.raises(ValueError, match=message):
        SplitFile.model_validate(raw)
