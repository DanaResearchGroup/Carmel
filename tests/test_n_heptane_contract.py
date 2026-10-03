"""Network-free frozen-contract, integrity, and chemical-class boundary checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jsonschema
import pytest
import yaml
from pydantic import ValidationError

from carmel.benchmarks.n_heptane import (
    Contract,
    Pin,
    Split,
    SplitFile,
    Study,
    git_blob,
    load_contract,
    load_split,
    split_totals,
    write_schema,
)
from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason
from carmel.services.chemked_archive import _MAX_FILE_BYTES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "benchmarks/n-heptane-low-t/contract.json"


@pytest.fixture
def raw() -> dict:
    return json.loads(CONTRACT.read_text())


def test_contract_schema_and_frozen_totals(tmp_path: Path, raw: dict) -> None:
    schema = json.loads(CONTRACT.with_name("schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(raw, schema)
    generated = tmp_path / "schema.json"
    write_schema(generated)
    assert json.loads(generated.read_text()) == schema
    contract = load_contract(CONTRACT)
    assert split_totals(contract.split) == {
        "development_files": 45,
        "development_points": 521,
        "holdout_files": 24,
        "holdout_points": 189,
        "holdout_below_750": 26,
        "holdout_from_750_to_900": 63,
        "holdout_rcm_ideal": 15,
    }
    assert [g.member_count for g in contract.ground_truth] == [77, 77, 26, 26, 34, 4]
    assert all(g.member_count == g.resolved_count for g in contract.ground_truth)
    assert {s.id for s in contract.split.studies if any(f.rcm_ideal_points for f in s.files)} == {
        "Di Sante 2012",
        "Karwat 2013",
    }
    assert contract.identity.status == "UNVERIFIED"
    assert contract.scoring.primary_unit == "reaction-type-once"
    assert "G6" not in contract.scoring.primary_types
    assert "minus baseline" in contract.scoring.baseline_present


def test_contract_rejects_overlapping_temperature_bins(raw: dict) -> None:
    item = raw["split"]["studies"][0]["files"][0]
    item["points"] = 1
    item["below_750"] = 1
    item["from_750_to_900"] = 1

    with pytest.raises(ValidationError, match="temperature bins"):
        SplitFile.model_validate(item)


def test_contract_mapping_fields_are_immutable_and_round_trip_unchanged(raw: dict) -> None:
    contract = Contract.model_validate(raw)

    with pytest.raises(TypeError):
        contract.mechanisms["v_old"] = contract.mechanisms["v_new"]
    mechanisms = contract.mechanisms
    with pytest.raises(TypeError):
        mechanisms |= {"new": contract.mechanisms["v_new"]}
    with pytest.raises(TypeError):
        contract.new_species[0].formula["C"] = 999
    with pytest.raises(TypeError):
        contract.ground_truth[0].reactant_formula["C"] = 999

    serialized = json.loads(contract.model_dump_json())
    assert serialized == raw
    assert json.loads(Contract.model_validate_json(contract.model_dump_json()).model_dump_json()) == raw


def test_contract_text_keeps_spaces_in_frozen_locking_phrases() -> None:
    text = CONTRACT.read_text()

    assert "denominator 4" in text
    assert "the 15 RCM points" in text


@pytest.mark.parametrize("change", ["zero", "one", "swapped"])
def test_contract_requires_ordered_glossary_and_thermo_pins(raw: dict, change: str) -> None:
    if change == "zero":
        raw["support_sources"] = []
    elif change == "one":
        raw["support_sources"] = raw["support_sources"][:1]
    else:
        raw["support_sources"].reverse()

    with pytest.raises(ValidationError, match="support sources must be glossary then thermo"):
        Contract.model_validate(raw)


def test_contract_rejects_ground_truth_reactant_formula_drift(raw: dict) -> None:
    raw["ground_truth"][0]["reactant_formula"] = {"C": 1, "H": 2}

    with pytest.raises(ValidationError, match="reactant formula"):
        Contract.model_validate(raw)


def test_reproduction_pin_fetcher_is_injectable(tmp_path: Path) -> None:
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location("n_heptane_reproduce", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"offline pinned input"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    calls: list[str] = []
    path = module.acquire_pinned(pin, tmp_path, fetcher=lambda url: calls.append(url) or data)
    assert path.read_bytes() == data
    assert calls == [pin.source_url]


def test_reproduction_uses_cached_pin_and_existing_expected_input(tmp_path: Path) -> None:
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location("n_heptane_reproduce_cached", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"cached pinned input"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    directory = tmp_path / "cache"
    directory.mkdir()
    cached = directory / pin.filename
    cached.write_bytes(data)
    assert module.acquire_pinned(pin, directory, fetcher=lambda _: pytest.fail("must use cache")) == cached
    expected = tmp_path / "expected.bin"
    expected.write_bytes(data)
    snapshot = module.input_path(pin, expected, None)
    assert snapshot.path == expected
    assert snapshot.data == data


def test_reproduction_fetch_url_and_missing_download_directory_are_testable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location("n_heptane_reproduce_fetch", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            assert limit > 0
            return b"fetched"

    monkeypatch.setattr(module, "urlopen", lambda url, timeout: Response())
    assert module.fetch_url("https://example.test/input.bin") == b"fetched"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(b"fetched").hexdigest(),
        bytes=len(b"fetched"),
    )
    with pytest.raises(FileNotFoundError, match="provide --download-dir"):
        module.input_path(pin, tmp_path / "missing.bin", None)


def test_reproduction_missing_input_uses_download_directory(tmp_path: Path) -> None:
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location("n_heptane_reproduce_download", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"downloaded pinned input"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    download_dir = tmp_path / "downloads"
    module._acquire_pinned_snapshot = lambda requested, directory: module.PinnedSnapshot(  # type: ignore[method-assign]
        download_dir / requested.filename, data
    )
    snapshot = module.input_path(pin, tmp_path / "missing.bin", download_dir)
    assert snapshot.path == download_dir / pin.filename
    assert snapshot.data == data


def test_reproduction_cached_oversize_is_rejected_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_reproduction("n_heptane_reproduce_cached_oversize")
    data = b"valid"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    cached = tmp_path / pin.filename
    cached.write_bytes(data + b"overflow")

    def fail_full_read(_: Path) -> bytes:
        raise AssertionError("an oversized cached pin must not be read in full")

    monkeypatch.setattr(Path, "read_bytes", fail_full_read)
    with pytest.raises(module.PinnedInputError, match="exceeds frozen byte count"):
        module.acquire_pinned(pin, tmp_path, fetcher=lambda _: pytest.fail("must use cache"))


def test_reproduction_refuses_a_target_symlink_created_after_validation(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_publish_race")
    data = b"downloaded"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"safe")
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )

    def fetch_and_create_symlink(_: str) -> bytes:
        (downloads / pin.filename).symlink_to(outside)
        return data

    with pytest.raises(module.PinnedInputError, match="appeared after validation"):
        module.acquire_pinned(pin, downloads, fetcher=fetch_and_create_symlink)
    assert outside.read_bytes() == b"safe"


def test_reproduction_parses_the_verified_input_snapshot_once(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_snapshot")
    original = b"SPECIES OLD\nEND\n"
    replacement = b"SPECIES NEW\nEND\n"
    pin = Pin(
        filename="mechanism.txt",
        source_url="https://example.test/mechanism.txt",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(original).hexdigest(),
        bytes=len(original),
    )
    path = tmp_path / pin.filename
    path.write_bytes(original)

    snapshot = module.input_path(pin, path, None)
    path.write_bytes(replacement)

    labels, _ = module.mechanism(snapshot.data)
    assert labels == ["OLD"]


def test_reproduction_rejects_ground_truth_reactant_formula_drift() -> None:
    module = _load_reproduction("n_heptane_reproduce_formula")
    ground_truth = SimpleNamespace(id="G1", reactant_formula={"C": 1, "H": 2})

    with pytest.raises(module.ContractMismatch, match="reactant formula"):
        module.validate_ground_truth_formula(ground_truth)


def test_reproduction_rejects_unparsed_reaction(tmp_path: Path) -> None:
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location("n_heptane_reproduce_mechanism", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "mechanism.txt"
    path.write_text("REACTIONS\nnot a parsed reaction = here\nEND\n")
    with pytest.raises(ValueError, match="unparsed reaction"):
        module.mechanism(path)


def _load_reproduction(name: str):
    script = ROOT / "benchmarks/n-heptane-low-t/reproduce.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduction_mechanism_parses_species_coefficients_and_comments(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_mechanism_full")
    path = tmp_path / "mechanism.txt"
    path.write_text(
        """
! header
SPECIES A B
C
END
REACTIONS
A + B => C 1.0E+00 0
2A + M <=> B + C 2.0D+00 0
ignored line
END
"""
    )
    labels, reactions = module.mechanism(path)
    assert labels == ["A", "B", "C"]
    assert reactions == [
        {"equation": "A+B=>C", "line": 7, "reactants": ("A", "B"), "products": ("C",)},
        {"equation": "2A+M<=>B+C", "line": 8, "reactants": ("A", "A"), "products": ("B", "C")},
    ]


def test_reproduction_glossary_handles_pdf_rows_and_invalid_smiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_reproduction("n_heptane_reproduce_glossary")
    structure = ModuleType("carmel.benchmarks.structure")

    def canonical(smiles: str) -> str:
        if smiles == "bad":
            raise ValueError("invalid")
        return f"canonical:{smiles}"

    structure.canonical = canonical  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carmel.benchmarks.structure", structure)
    (tmp_path / "glossary.pdf").write_bytes(b"pdf")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *args, **kwargs: "A C7H16 raw InChI=1S/a\nB H2 bad InChI=1S/b\nignored row\nC C1H2 raw no-inchi\n",
    )
    rows = module.glossary(tmp_path / "glossary.pdf")
    assert rows == {
        "A": {"smiles": "canonical:raw", "formula": {"C": 7, "H": 16}, "source": "author-glossary"},
        "B": {"smiles": None, "formula": {"H": 2}, "source": "author-glossary"},
    }


def test_reproduction_thermo_reads_fixed_width_formula_rows(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_thermo")
    valid = [" "] * 80
    valid[:18] = list("SPECIES".ljust(18))
    valid[24:27] = list("C 7")
    valid[29:34] = list("H  16")
    valid[34:39] = list("O   0")
    valid[79] = "1"
    invalid = "ignored".ljust(79) + "0"
    path = tmp_path / "thermo.txt"
    path.write_text("".join(valid) + "\n" + invalid + "\n")
    assert module.thermo(path) == {"SPECIES": {"C": 7, "H": 16}}


def test_reproduction_inventory_uses_grammar_and_formula_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_reproduction("n_heptane_reproduce_inventory")
    structure = ModuleType("carmel.benchmarks.structure")

    def features(smiles: str) -> SimpleNamespace:
        return SimpleNamespace(atoms={"C": {"A": 1, "K": 2, "M": 9}[smiles]})

    structure.features = features  # type: ignore[attr-defined]
    structure.ketohydroperoxide = lambda label: "K" if label == "N1" else None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carmel.benchmarks.structure", structure)
    dictionary = {
        "A": {"smiles": "A", "source": "author-glossary"},
        "1": {"smiles": "K", "source": "author-glossary"},
        "N1": {"smiles": None, "source": "author-glossary"},
        "M": {"smiles": "M", "source": "author-glossary"},
    }
    assert module.inventory(["A"], dictionary) == [
        {"label": "A", "status": "RESOLVED", "smiles": "A", "formula": {"C": 1}, "source": "author-glossary"}
    ]
    records = module.inventory(
        ["N1", "missing", "M"],
        dictionary,
        {"N1": {"C": 2}, "missing": {"C": 3}, "M": {"C": 10}},
    )
    assert records == [
        {
            "label": "N1",
            "status": "RESOLVED",
            "smiles": "K",
            "formula": {"C": 2},
            "source": "tested-label-grammar: N?CnKETij",
        },
        {
            "label": "missing",
            "status": "UNRESOLVED",
            "smiles": None,
            "formula": {"C": 3},
            "source": "LLNL-thermo: formula only; no confident structure",
        },
        {
            "label": "M",
            "status": "UNRESOLVED",
            "smiles": None,
            "formula": {"C": 10},
            "source": "LLNL-thermo: formula only; no confident structure",
        },
    ]


def test_reproduction_candidate_mapping_and_frozen_split_are_deterministic(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_mapping")
    species = {
        "match": {"formula": {"C": 7, "H": 15, "O": 4}},
        "other": {"formula": {"C": 1}},
    }
    reactions = [
        {"reactants": ("match",), "products": ("other",)},
        {"reactants": ("other",), "products": ("other",)},
    ]
    assert list(module.candidate_reactions(reactions, species)) == [reactions[0]]

    raw = tmp_path / "source.json"
    census = tmp_path / "census"
    file_path = census / "chemked" / "study" / "one.yaml"
    file_path.parent.mkdir(parents=True)
    content = (
        b"apparatus:\n  kind: rapid compression machine\n"
        b"datapoints:\n  - temperature: ['700 K']\n  - temperature: ['800 K']\n"
    )
    file_path.write_bytes(content)
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    (census / "evidence").mkdir()
    (census / "evidence/chemked-local-ledger.json").write_text(
        json.dumps([{"path": "study/one.yaml", "git_blob": digest}])
    )
    raw.write_text(
        json.dumps(
            {
                "source_commit": "abc123",
                "created_utc": "2026-10-02T00:00:00Z",
                "studies": [
                    {
                        "study": "Study",
                        "doi": "doi:1",
                        "assignment": "development",
                        "points": 2,
                        "paths": ["chemked/study/one.yaml"],
                    }
                ],
            }
        )
    )
    result = module.frozen_split(raw, census)
    assert result["commit"] == "abc123"
    assert result["studies"][0]["files"][0]["below_750"] == 1
    assert result["studies"][0]["files"][0]["rcm_ideal_points"] == [0, 1]


def test_reproduction_frozen_split_uses_bounded_local_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_reproduction("n_heptane_reproduce_bounded_split")
    raw = tmp_path / "source.json"
    census = tmp_path / "census"
    file_path = census / "chemked" / "study" / "one.yaml"
    file_path.parent.mkdir(parents=True)
    content = b"apparatus:\n  kind: shock tube\ndatapoints:\n  - temperature: ['700 K']\n"
    file_path.write_bytes(content)
    digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    (census / "evidence").mkdir()
    (census / "evidence/chemked-local-ledger.json").write_text(
        json.dumps([{"path": "study/one.yaml", "git_blob": digest}])
    )
    raw.write_text(
        json.dumps(
            {
                "source_commit": "abc123",
                "created_utc": "2026-10-02T00:00:00Z",
                "studies": [
                    {
                        "study": "Study",
                        "doi": "doi:1",
                        "assignment": "development",
                        "points": 1,
                        "paths": ["chemked/study/one.yaml"],
                    }
                ],
            }
        )
    )
    calls: list[Path] = []
    reader = module._read_pinned_root_file

    def bounded_reader(path: Path) -> bytes:
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(module, "_read_pinned_root_file", bounded_reader)
    module.frozen_split(raw, census)
    assert calls == [raw, census / "evidence/chemked-local-ledger.json", file_path]


def test_reproduction_check_and_map_records_structural_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_reproduction("n_heptane_reproduce_check")
    structure = ModuleType("carmel.benchmarks.structure")
    structure.features = lambda smiles: SimpleNamespace(  # type: ignore[attr-defined]
        atoms={"C": 7, "H": 15, "O": 4} if smiles == "new-a" else {"C": 1}
    )
    structure.classify = lambda left, right: ["G1"]  # type: ignore[attr-defined]
    structure.reaction_signature = lambda left, right: (left, right)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carmel.benchmarks.structure", structure)
    old_species = [
        {"label": "old-a", "formula": {"C": 7, "H": 15, "O": 4}, "status": "RESOLVED", "smiles": "old-a"},
        {"label": "old-b", "formula": {"C": 1}, "status": "RESOLVED", "smiles": "old-b"},
    ]
    old_reactions = [{"reactants": ("old-a",), "products": ("old-b",)}]
    new_reactions = [{"reactants": ("new-a",), "products": ("new-b",)}]
    new_dictionary = {
        "new-a": {"smiles": "new-a", "formula": {"C": 7, "H": 15, "O": 4}},
        "new-b": {"smiles": "new-b", "formula": {"C": 1}},
    }
    members, signatures = module.check_and_map(old_reactions, new_reactions, old_species, new_dictionary)
    assert signatures == 1
    assert members["G1"][0]["absence"] == "structural-signature-and-exhaustive-formula-screen"


def test_reproduction_main_wires_the_frozen_audit_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_reproduction("n_heptane_reproduce_main")

    class PinStub:
        def __init__(self, filename: str, unique_species: int = 1, reaction_entries: int = 0) -> None:
            self.filename = filename
            self.unique_species = unique_species
            self.reaction_entries = reaction_entries

        def verify(self, data: bytes) -> None:
            assert data in {b"old", b"new", b"pinned"}

    old_pin = PinStub("old.txt", unique_species=1)
    new_pin = PinStub("new.txt", unique_species=1)
    glossary_pin = PinStub("glossary.pdf")
    thermo_pin = PinStub("thermo.txt")
    old_inventory = [{"label": "old", "status": "UNRESOLVED", "smiles": None, "formula": {"C": 1}, "source": "source"}]
    new_inventory = [{"label": "new", "status": "UNRESOLVED", "smiles": None, "formula": {"C": 1}, "source": "source"}]
    ground_truth = SimpleNamespace(
        id="G1",
        reactant_formula={"C": 7, "H": 15, "O": 4},
        member_count=1,
        resolved_count=1,
        resolved_species_count=1,
        member_species_count=1,
        members=[SimpleNamespace(model_dump=lambda mode=None: {"entry": 1})],
    )
    contract = SimpleNamespace(
        mechanisms={"v_old": old_pin, "v_new": new_pin},
        glossary_source=glossary_pin,
        thermo_source=thermo_pin,
        old_species=[SimpleNamespace(model_dump=lambda: old_inventory[0])],
        new_species=[SimpleNamespace(label="new", model_dump=lambda: new_inventory[0])],
        ground_truth=[ground_truth],
        split=SimpleNamespace(model_dump=lambda mode=None: {"split": True}),
    )
    census = tmp_path / "census"
    support = tmp_path / "support"
    (census / "evidence").mkdir(parents=True)
    support.mkdir()

    def fake_input_path(pin, expected, download_directory):
        expected.parent.mkdir(parents=True, exist_ok=True)
        data = b"old" if pin is old_pin else b"new" if pin is new_pin else b"pinned"
        expected.write_bytes(data)
        return module.PinnedSnapshot(expected, data)

    monkeypatch.setattr(module, "load_contract", lambda _: contract)
    monkeypatch.setattr(module, "input_path", fake_input_path)
    monkeypatch.setattr(
        module,
        "mechanism",
        lambda data: (["old"], []) if data == b"old" else (["new"], []),
    )
    monkeypatch.setattr(module, "glossary", lambda path: {"new": {"smiles": None, "formula": {"C": 1}}})
    monkeypatch.setattr(module, "thermo", lambda path: {})
    monkeypatch.setattr(
        module,
        "inventory",
        lambda labels, dictionary, formulas=None: old_inventory if labels == ["old"] else new_inventory,
    )
    monkeypatch.setattr(module, "check_and_map", lambda *args: ({"G1": [{"entry": 1}]}, 0))
    monkeypatch.setattr(module, "frozen_split", lambda split_report, census_root: {"split": True})
    monkeypatch.setattr(module, "load_split", lambda contract, root: {"development": 1})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "reproduce.py",
            "--census-root",
            str(census),
            "--support-root",
            str(support),
            "--split-report",
            str(tmp_path / "split.json"),
        ],
    )

    module.main()
    output = capsys.readouterr().out
    assert "PASS G1" in output
    assert "PASS split" in output
    assert "IDENTITY: UNVERIFIED" in output


def test_reproduction_live_download_is_bounded_before_pin_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_reproduction("n_heptane_reproduce_bounded_fetch")
    data = b"abc"
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data),
    )
    reads: list[int] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            reads.append(limit)
            return b"x" * limit

    monkeypatch.setattr(module, "urlopen", lambda url, timeout: Response())
    with pytest.raises(module.PinnedInputError, match="exceeds frozen byte count"):
        module.acquire_pinned(pin, tmp_path, fetcher=module.fetch_url)
    assert reads == [pin.bytes + 1]


@pytest.mark.parametrize("filename", ["../input.bin", "/tmp/input.bin", "nested/input.bin", ".."])
def test_reproduction_rejects_untrusted_pin_filenames_before_io(tmp_path: Path, filename: str) -> None:
    module = _load_reproduction("n_heptane_reproduce_filename")
    pin = Pin(
        filename=filename,
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(b"data").hexdigest(),
        bytes=4,
    )
    with pytest.raises(module.PinnedInputError, match="plain basename"):
        module.acquire_pinned(pin, tmp_path, fetcher=lambda _: pytest.fail("must reject before fetching"))


def test_reproduction_rejects_a_cached_symlink_escaping_download_dir(tmp_path: Path) -> None:
    module = _load_reproduction("n_heptane_reproduce_symlink")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"data")
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "input.bin").symlink_to(outside)
    pin = Pin(
        filename="input.bin",
        source_url="https://example.test/input.bin",
        retrieval_date="2026-10-02",
        sha256=hashlib.sha256(b"data").hexdigest(),
        bytes=4,
    )
    with pytest.raises(module.PinnedInputError, match="contained"):
        module.acquire_pinned(pin, downloads, fetcher=lambda _: pytest.fail("must reject before reading"))


def test_reproduction_integrity_gates_survive_python_optimization() -> None:
    source = (ROOT / "benchmarks/n-heptane-low-t/reproduce.py").read_text()
    assert "assert " not in source


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing_pin", "both mechanism pins"),
        ("missing_type", "six reviewed types"),
        ("extra_primary", "never be scored against ignition"),
        ("duplicate_species", "species labels must be unique"),
        ("duplicate_old", "species labels must be unique"),
        ("old_missing", "inventory every old species"),
        ("unknown_member", "uninventoried species"),
        ("wrong_count", "counts do not reproduce"),
        ("scope", "speciation-only"),
        ("low_resolution", "fewer than 70%"),
        ("duplicate_entry", "entry lines must be unique"),
        ("duplicate_path", "must be disjoint"),
        ("duplicate_study", "must be disjoint"),
        ("duplicate_doi", "DOI must not cross"),
        ("wrong_split", "split counts do not reproduce"),
        ("zhang_missing", "Zhang2016 data must stay"),
    ],
)
def test_contract_rejects_drift(raw: dict, change: str, error: str) -> None:
    gt = raw["ground_truth"][0]
    studies = raw["split"]["studies"]
    if change == "missing_pin":
        raw["mechanisms"].pop("v_old")
    elif change == "missing_type":
        raw["ground_truth"].pop()
    elif change == "extra_primary":
        raw["scoring"]["primary_types"].append("G6")
    elif change == "duplicate_species":
        raw["new_species"].append(raw["new_species"][0])
    elif change == "duplicate_old":
        raw["old_species"].append(raw["old_species"][0])
    elif change == "old_missing":
        raw["old_species"].pop()
    elif change == "unknown_member":
        gt["members"][0]["reactants"] = ["UNKNOWN"]
    elif change == "wrong_count":
        gt["resolved_count"] -= 1
    elif change == "scope":
        raw["ground_truth"][-1]["scope"] = "primary"
    elif change == "low_resolution":
        labels = {s for m in gt["members"] for s in (*m["reactants"], *m["products"])}
        for species in raw["new_species"]:
            if species["label"] in labels:
                species.update(status="UNRESOLVED", smiles=None)
        gt.update(resolved_count=0, resolved_species_count=0)
    elif change == "duplicate_entry":
        gt["members"][1]["line"] = gt["members"][0]["line"]
    elif change == "duplicate_path":
        studies[0]["files"][1]["path"] = studies[0]["files"][0]["path"]
    elif change == "duplicate_study":
        studies[1]["id"] = studies[0]["id"]
    elif change == "duplicate_doi":
        studies[2]["doi"] = studies[1]["doi"]
    elif change == "wrong_split":
        studies[0]["files"][0]["points"] += 1
    elif change == "zhang_missing":
        studies[-1]["id"] = "Zhang renamed"
    with pytest.raises(ValidationError, match=error):
        Contract.model_validate(raw)


@pytest.mark.parametrize(
    "change",
    ["absolute", "traversal", "backslash_traversal", "outside", "duplicate_index", "large_index", "negative_index"],
)
def test_split_file_rejects_unsafe_paths_or_indices(raw: dict, change: str) -> None:
    item = raw["split"]["studies"][0]["files"][0]
    if change == "absolute":
        item["path"] = "/n-heptane/example.yaml"
    elif change == "traversal":
        item["path"] = "n-heptane/../other.yaml"
    elif change == "backslash_traversal":
        item["path"] = r"n-heptane/..\..\outside.yaml"
    elif change == "outside":
        item["path"] = "other/example.yaml"
    elif change == "duplicate_index":
        item["rcm_ideal_points"] = [0, 0]
    elif change == "large_index":
        item["rcm_ideal_points"] = [item["points"]]
    else:
        item["rcm_ideal_points"] = [-1]
    with pytest.raises(ValidationError):
        SplitFile.model_validate(item)


@pytest.mark.parametrize("status,smiles", [("UNRESOLVED", "CCCCCCC"), ("RESOLVED", None)])
def test_unresolved_is_never_guessed(raw: dict, status: str, smiles: str | None) -> None:
    raw["new_species"][0].update(status=status, smiles=smiles)
    with pytest.raises(ValidationError, match="guessed structure"):
        Contract.model_validate(raw)


def tiny_split(data: dict) -> tuple[Contract, bytes, SplitFile]:
    """Synthetic single-file split isolates loader behavior, without third-party YAML."""
    raw = yaml.safe_dump(data).encode()
    item = SplitFile(
        path="n-heptane/synthetic/test.yaml",
        git_blob=git_blob(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        points=1,
        below_750=1,
        from_750_to_900=0,
        rcm_ideal_points=(0,) if data["apparatus"]["kind"] == "rapid compression machine" else (),
    )
    original = load_contract(CONTRACT)
    split = Split(
        repository=original.split.repository,
        commit=original.split.commit,
        frozen_at=original.split.frozen_at,
        studies=(Study(id="synthetic", doi=None, set="holdout", files=(item,)),),
    )
    # model_copy deliberately avoids the real-contract total gate for this focused loader fixture.
    return original.model_copy(update={"split": split}), raw, item


@pytest.fixture
def point_data() -> dict:
    return {"apparatus": {"kind": "shock tube"}, "datapoints": [{"temperature": ["700 K"], "ignition-delay": ["1 ms"]}]}


def test_loader_verifies_local_and_injected_bytes(tmp_path: Path, point_data: dict) -> None:
    contract, data, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    assert load_split(contract, tmp_path)["holdout_points"] == 1
    assert load_split(contract, fetcher=lambda _: data)["holdout_below_750"] == 1
    with pytest.raises(ValueError, match="SPLIT BLOB CHANGED"):
        load_split(contract, fetcher=lambda _: data + b"\n")
    point_data["apparatus"]["kind"] = "rapid compression machine"
    contract, data, _ = tiny_split(point_data)
    assert load_split(contract, fetcher=lambda _: data)["holdout_rcm_ideal"] == 1


def test_loader_rejects_oversized_local_file_before_read(
    tmp_path: Path, point_data: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * (_MAX_FILE_BYTES + 1))

    def fail_read(_: Path) -> bytes:
        raise AssertionError("an oversized local file must not be read")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(ValueError, match="over the .* cap"):
        load_split(contract, tmp_path)


def test_loader_rejects_local_file_replaced_during_read(
    tmp_path: Path, point_data: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, data, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    real_stat = Path.stat
    calls = 0

    def changing_stat(candidate: Path) -> object:
        nonlocal calls
        if candidate == path:
            calls += 1
            return SimpleNamespace(st_size=len(data), st_dev=1, st_ino=1 if calls == 1 else 2)
        return real_stat(candidate)

    monkeypatch.setattr(Path, "stat", changing_stat)
    with pytest.raises(ValueError, match="changed while reading"):
        load_split(contract, tmp_path)


def test_loader_rejects_file_that_grows_during_read(
    tmp_path: Path, point_data: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, data, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    real_open = Path.open

    def growing_open(candidate: Path, *args: object, **kwargs: object):
        stream = real_open(candidate, *args, **kwargs)
        if candidate != path:
            return stream

        class GrowingStream:
            def __enter__(self) -> GrowingStream:
                stream.__enter__()
                return self

            def __exit__(self, *exit_args: object) -> None:
                stream.__exit__(*exit_args)

            def read(self, limit: int) -> bytes:
                assert limit == _MAX_FILE_BYTES + 1
                return b"x" * limit

        return GrowingStream()

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="over the .* cap"):
        load_split(contract, tmp_path)


def test_loader_reports_local_stat_failure(tmp_path: Path, point_data: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, _, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    real_stat = Path.stat

    def failing_stat(candidate: Path) -> object:
        if candidate == path:
            raise OSError("stat failed")
        return real_stat(candidate)

    monkeypatch.setattr(Path, "stat", failing_stat)
    with pytest.raises(ValueError, match="cannot read local ChemKED file"):
        load_split(contract, tmp_path)


def test_loader_reports_local_read_failure(tmp_path: Path, point_data: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, data, item = tiny_split(point_data)
    path = tmp_path / item.path
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    real_open = Path.open

    def failing_open(candidate: Path, *args: object, **kwargs: object):
        if candidate == path:
            raise OSError("read failed")
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(ValueError, match="cannot read local ChemKED file"):
        load_split(contract, tmp_path)


def test_loader_rejects_unknown_apparatus_with_typed_error(point_data: dict) -> None:
    point_data["apparatus"]["kind"] = "closed vessel"
    contract, data, _ = tiny_split(point_data)

    with pytest.raises(ChemkedRefusal) as caught:
        load_split(contract, fetcher=lambda _: data)
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_APPARATUS


@pytest.mark.parametrize("change", ["unit", "missing_idt", "volume", "time_history", "counts"])
def test_loader_rejects_invalid_point_metadata(point_data: dict, change: str) -> None:
    point = point_data["datapoints"][0]
    if change == "unit":
        point["temperature"] = ["700 degC"]
    elif change == "missing_idt":
        point.pop("ignition-delay")
    elif change in {"volume", "time_history"}:
        point_data["apparatus"]["kind"] = "rapid compression machine"
        if change == "volume":
            point["volume-history"] = {}
        else:
            point["time-histories"] = [{"type": "volume"}]
    elif change == "counts":
        point["temperature"] = ["900 K"]
    contract, data, _ = tiny_split(point_data)
    with pytest.raises(ValueError):
        load_split(contract, fetcher=lambda _: data)


def test_loader_uses_carmel_manifest_and_cache(monkeypatch: pytest.MonkeyPatch, point_data: dict) -> None:
    from carmel.services import chemked_archive as archive

    contract, data, item = tiny_split(point_data)
    pinned = archive.ChemkedFile(path=item.path, sha256=item.sha256)
    manifest = archive.ChemkedManifest(
        repository=contract.split.repository, commit=contract.split.commit, files=(pinned,)
    )
    monkeypatch.setattr(archive, "load_manifest", lambda: manifest)
    monkeypatch.setattr(archive, "fetch_file", lambda *args: data)
    assert load_split(contract)["holdout_points"] == 1
    manifest = archive.ChemkedManifest(repository="other/repository", commit=contract.split.commit, files=(pinned,))
    with pytest.raises(ValueError, match="repository differs"):
        load_split(contract)
    manifest = archive.ChemkedManifest(repository=contract.split.repository, commit="0" * 40, files=(pinned,))
    with pytest.raises(ValueError, match="commit differs"):
        load_split(contract)
    manifest = archive.ChemkedManifest(
        repository=contract.split.repository,
        commit=contract.split.commit,
        files=(archive.ChemkedFile(path=item.path, sha256="0" * 64),),
    )
    with pytest.raises(ValueError, match="pin differs"):
        load_split(contract)


def test_pin_bytes_fail_loudly(raw: dict) -> None:
    data = b"tiny synthetic mechanism"
    pin = deepcopy(raw["mechanisms"]["v_old"])
    pin.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    raw["mechanisms"]["v_old"] = pin
    contract = Contract.model_validate(raw)
    contract.mechanisms["v_old"].verify(data)
    with pytest.raises(ValueError, match="PIN CHANGED"):
        contract.mechanisms["v_old"].verify(data + b"\n")
    with pytest.raises(ValueError, match="PIN CHANGED"):
        contract.mechanisms["v_old"].verify(b"x" * len(data))
