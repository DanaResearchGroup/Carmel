"""Network-free coverage of the case-2 reproducer's pins, member selection, split and CLI."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/n-pentane-low-t/contract.json"
EXPECTED = {"G1": 30, "G2": 30, "G3": 12, "G4": 12}


@pytest.fixture
def reproducer() -> ModuleType:
    pytest.importorskip("rdkit")  # the reproducer imports the structure toolkit at load time
    spec = importlib.util.spec_from_file_location(
        "n_pentane_reproduce_cov", ROOT / "benchmarks/n-pentane-low-t/reproduce.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pin_records_the_hash_and_inventory(reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reproducer, "mechanism", lambda raw: (["A", "B"], [{}]))
    pinned = reproducer.pin(b"abc", filename="m.dat", source_url="https://example.org/m", retrieval_date="2026-11-07")
    assert pinned == {
        "filename": "m.dat",
        "source_url": "https://example.org/m",
        "retrieval_date": "2026-11-07",
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "bytes": 3,
        "unique_species": 2,
        "reaction_entries": 1,
    }
    with pytest.raises(ValueError, match="boom"):
        reproducer.require(False, "boom")


@pytest.mark.parametrize(
    "label,source",
    [
        ("C5H11O2-1", "tested Bugler/NUIG straight-chain C5 label grammar"),
        ("NC5KET24", "tested N?CnKETij grammar"),
        ("HO2", "explicit small-radical structure"),
    ],
)
def test_species_record_resolves_and_names_its_grammar(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch, label: str, source: str
) -> None:
    monkeypatch.setattr(reproducer, "pentane_low_temperature_species", lambda name: "OO")
    formula = dict(reproducer.features("OO").atoms)
    assert formula == {"O": 2, "H": 2}
    record = reproducer.species_record(label, formula)
    assert record == {"label": label, "status": "RESOLVED", "smiles": "OO", "formula": formula, "source": source}
    with pytest.raises(ValueError, match="structure formula conflicts with mechanism composition"):
        reproducer.species_record(label, {"O": 2})


def test_species_record_refuses_an_unresolved_member_but_records_an_unresolved_baseline(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reproducer, "pentane_low_temperature_species", lambda name: None)
    with pytest.raises(ValueError, match="UNRESOLVED member species: X"):
        reproducer.species_record("X", {"C": 1})
    record = reproducer.species_record("X", {"C": 1}, required=False)
    assert record["status"] == "UNRESOLVED" and record["smiles"] is None and record["formula"] == {"C": 1}


def _members(reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch, counts: dict[str, int], old_extra=()):
    # One synthetic reaction per expected member: reactant "<group>_<i>" -> product "P_<group>_<i>".
    new = [
        {"equation": f"{g}_{i}=P_{g}_{i}", "reactants": [f"{g}_{i}"], "products": [f"P_{g}_{i}"]}
        for g, n in counts.items()
        for i in range(n)
    ]
    new.append({"equation": "U=V", "reactants": ["U"], "products": ["V"]})  # V has no structure: skipped
    new.append({"equation": "Q=R", "reactants": ["Q"], "products": ["R"]})  # no scored class: skipped
    old = [{"equation": "O1=O2", "reactants": ["O1"], "products": ["O2"]}, *old_extra]
    extra_labels = sorted({n for r in old_extra for n in (*r["reactants"], *r["products"])})
    sources = {"old": (["O1", "O2", "Z", *extra_labels], old), "new": ([], new)}
    absence_calls = []
    monkeypatch.setattr(reproducer, "mechanism", lambda raw: sources[raw.decode()])
    monkeypatch.setattr(
        reproducer, "formula_inventory", lambda labels, reactions: {label: {"C": 1} for label in labels}
    )
    monkeypatch.setattr(
        reproducer, "pentane_low_temperature_species", lambda name: None if name in {"V", "Z"} else name
    )
    monkeypatch.setattr(reproducer, "features", lambda smiles: SimpleNamespace(atoms={"C": 1}))
    monkeypatch.setattr(reproducer, "reaction_signature", lambda left, right: (left, right))
    monkeypatch.setattr(
        reproducer, "classify", lambda left, right, carbon_count: {left[0].split("_")[0]} if "_" in left[0] else set()
    )
    monkeypatch.setattr(reproducer, "require_structural_absence", lambda *args: absence_calls.append(args[0]))
    return absence_calls


def test_mapped_ground_truth_groups_new_members_and_audits_absence(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    absence_calls = _members(reproducer, monkeypatch, EXPECTED)
    new_species, old_species, truth = reproducer.mapped_ground_truth(b"old", b"new")
    assert [t["id"] for t in truth] == ["G1", "G2", "G3", "G4"]
    assert {t["id"]: t["member_count"] for t in truth} == EXPECTED
    assert all(t["member_species_count"] == 2 * t["member_count"] for t in truth)
    assert all(
        m["absence"] == "structural-signature-and-exhaustive-formula-screen" for t in truth for m in t["members"]
    )
    assert len(absence_calls) == sum(EXPECTED.values())
    assert len(new_species) == 2 * sum(EXPECTED.values())
    assert [(s["label"], s["status"]) for s in old_species] == [
        ("O1", "RESOLVED"),
        ("O2", "RESOLVED"),
        ("Z", "UNRESOLVED"),
    ]


def test_mapped_ground_truth_refuses_count_drift_or_a_baseline_signature(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _members(reproducer, monkeypatch, {**EXPECTED, "G4": 11})
    with pytest.raises(ValueError, match="member counts changed"):
        reproducer.mapped_ground_truth(b"old", b"new")
    prior = {"equation": "G1_0=P_G1_0", "reactants": ["G1_0"], "products": ["P_G1_0"]}
    _members(reproducer, monkeypatch, EXPECTED, old_extra=[prior])
    with pytest.raises(ValueError, match="baseline reaction signature already present: G1_0=P_G1_0"):
        reproducer.mapped_ground_truth(b"old", b"new")


def _xml(doi: str, fuel: str = "NC5H12") -> bytes:
    return (
        f"<experiment><bibliographyLink><referenceDOI>doi.org/{doi}</referenceDOI></bibliographyLink>"
        f'<commonProperties><property name="initial composition"><component><speciesLink preferredKey="{fuel}"/>'
        "</component></property></commonProperties>"
        '<dataGroup><property name="temperature" id="x1"/><property name="ignition delay" id="x2"/>'
        "<dataPoint><x1>700</x1><x2>1</x2></dataPoint><dataPoint><x1>800</x1></dataPoint></dataGroup>"
        '<dataGroup><property name="temperature" id="x3"/><dataPoint><x3>900</x3></dataPoint></dataGroup>'
        "</experiment>"
    ).encode()


def test_frozen_split_classifies_and_counts_each_study_file(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a_dev.xml", _xml(reproducer.DEVELOPMENT_DOI))
        archive.writestr("b_hold.xml", _xml(reproducer.HOLDOUT_DOI))
        archive.writestr("c_other.xml", _xml("10.1/elsewhere"))
        archive.writestr("d_iso.xml", _xml(reproducer.HOLDOUT_DOI, fuel="IC5H12"))
        archive.writestr("readme.txt", "not xml")
    raw = buffer.getvalue()

    def refuse(raw: bytes, archive: object, name: str) -> object:
        raise reproducer.RespecthRefusal(_reason(), "no")

    monkeypatch.setattr(reproducer, "parse_idt_record", lambda raw, archive, name: ("record", archive.sha256))
    monkeypatch.setattr(reproducer, "parse_series_record", refuse)
    monkeypatch.setattr(reproducer, "series_counts", lambda record: (0, 0, 1, 2))
    split = reproducer.frozen_split(raw)
    assert split["archive"]["bytes"] == len(raw)
    development, holdout = (study["files"] for study in split["studies"])
    assert [f["path"] for f in development] == ["a_dev.xml"]
    assert development[0] | {"sha256": None} == {
        "path": "a_dev.xml",
        "sha256": None,
        "rows": 2,
        "observations": 1,
        "low_temperature_rows": 1,
        "low_temperature_observations": 2,
        "acceptance": "supported",
        "refusal_reason": None,
    }
    assert [(f["path"], f["acceptance"], f["refusal_reason"]) for f in holdout] == [
        ("b_hold.xml", "refused", _reason().value)
    ]
    assert holdout[0]["low_temperature_rows"] == 0


def _reason():
    from carmel.services.respecth import RespecthRefusalReason

    return RespecthRefusalReason.INCOMPLETE_RECORD


def test_build_contract_assembles_pins_split_and_truth(reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reproducer, "mapped_ground_truth", lambda old, new: (["new"], ["old"], ["truth"]))
    monkeypatch.setattr(reproducer, "frozen_split", lambda raw: {"archive": {"filename": "a.zip"}, "studies": []})
    monkeypatch.setattr(reproducer, "mechanism", lambda raw: (["A"], [{}]))
    built = reproducer.build_contract(b"old", b"new", b"zip")
    assert built["mechanisms"]["v_old"]["filename"] == "nc5_49_mech.dat"
    assert built["mechanisms"]["v_new"]["filename"] == "nuigmech_c5_july2015_1215.chem"
    assert built["support_sources"] == [{"filename": "a.zip"}]
    assert (built["new_species"], built["old_species"], built["ground_truth"]) == (["new"], ["old"], ["truth"])


def _run_main(reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch, built: dict, *args: str) -> list[object]:
    verified: list[object] = []
    frozen = reproducer.load_contract(CONTRACT)
    for pinned in (frozen.mechanisms["v_old"], frozen.split.archive):
        monkeypatch.setattr(type(pinned), "verify", lambda self, raw: verified.append(raw))
    monkeypatch.setattr(reproducer, "read_pinned_file", lambda path: path.name.encode())
    monkeypatch.setattr(reproducer, "build_contract", lambda old, new, archive: built)
    monkeypatch.setattr(reproducer, "load_split", lambda contract, archive: {"rows": 1})
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "--v-old", "old", "--v-new", "new", "--archive", "a", *args])
    reproducer.main()
    return verified


def test_main_audits_the_frozen_contract_against_the_reproduction(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frozen = json.loads(CONTRACT.read_bytes())
    verified = _run_main(reproducer, monkeypatch, frozen, "--contract", str(CONTRACT))
    assert verified == [b"old", b"new", b"a"]
    output = capsys.readouterr().out
    assert output.count("structural absence verified") == len(frozen["ground_truth"])
    assert 'PASS split: {"rows": 1}' in output and "IDENTITY: UNVERIFIED" in output
    changed = {**frozen, "leakage_limits": [*frozen["leakage_limits"], "one more"]}
    with pytest.raises(ValueError, match="frozen contract changed"):
        _run_main(reproducer, monkeypatch, changed, "--contract", str(CONTRACT))


def test_main_writes_the_contract_byte_for_byte(
    reproducer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written = tmp_path / "nested/contract.json"
    verified = _run_main(
        reproducer, monkeypatch, json.loads(CONTRACT.read_bytes()), "--write-contract", "--contract", str(written)
    )
    assert verified == []
    assert written.read_bytes() == CONTRACT.read_bytes()
