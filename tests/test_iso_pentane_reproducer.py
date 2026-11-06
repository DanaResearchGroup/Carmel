"""Network-free coverage of the case-3 reproducer's pin, audit, conversion and build steps."""

import dataclasses
import hashlib
import io
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/iso-pentane-low-t/reproduce.py"
CONTRACT = ROOT / "benchmarks/iso-pentane-low-t/contract.json"
RECORD = (ROOT / "tests/data/reviewed_ec6_thermo.dat").read_bytes()


@pytest.fixture
def reproducer(monkeypatch):
    # The reproducer imports the RDKit-backed structure rules at load time.
    pytest.importorskip("rdkit")
    namespace = runpy.run_path(str(SCRIPT))
    module_globals = namespace["build"].__globals__

    def patch(**values):
        for name, value in values.items():
            monkeypatch.setitem(module_globals, name, value)

    return SimpleNamespace(g=module_globals, patch=patch)


def _pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_pinned_and_source_pin_accept_only_the_recorded_bytes(reproducer, tmp_path):
    g = reproducer.g
    path = tmp_path / "source.txt"
    path.write_bytes(b"pinned bytes")
    assert g["pinned"](path, _pin(b"pinned bytes")) == b"pinned bytes"
    with pytest.raises(ValueError, match="NEEDS-INPUT: PIN CHANGED"):
        g["pinned"](path, _pin(b"other bytes"))
    with pytest.raises(ValueError, match="PIN CHANGED"):
        g["pinned"](path, {"bytes": 1, "sha256": _pin(b"pinned bytes")["sha256"]})
    digest = _pin(b"raw")["sha256"]
    assert g["source_pin"]("f.txt", "https://example.org/f.txt", "2026-11-07", b"raw", digest) == {
        "filename": "f.txt",
        "source_url": "https://example.org/f.txt",
        "retrieval_date": "2026-11-07",
        "sha256": digest,
        "bytes": 3,
        "unique_species": None,
        "reaction_entries": None,
    }
    with pytest.raises(ValueError, match="PIN CHANGED: f.txt"):
        g["source_pin"]("f.txt", "u", "d", b"changed", digest)


def test_thermo_inventory_counts_records_and_refuses_conflicts_or_empty_input(reproducer):
    inventory = reproducer.g["thermo_inventory"]
    assert inventory(RECORD) == ({"EC6H13O2": {"C": 6, "H": 13, "O": 2}}, 1)
    # An identical duplicate is accepted and counted; the first record wins.
    assert inventory(RECORD + RECORD)[1] == 2
    conflicting = RECORD.replace(b"O   2    0G", b"O   3    0G")
    assert conflicting != RECORD
    with pytest.raises(ValueError, match="conflicting formula: EC6H13O2"):
        inventory(RECORD + conflicting)
    with pytest.raises(ValueError, match="empty thermo inventory"):
        inventory(b"THERMO\nEND\n")


def test_nasa_preservation_audit_compares_every_record_field(reproducer):
    audit = reproducer.g["audit_nasa_preservation"]
    # Changes after the fixed-width NASA fields (comments, line endings) are allowed.
    assert audit(RECORD, RECORD.replace(b"\r\n", b"\n")) == 1
    lines = RECORD.splitlines(keepends=True)
    coefficient = lines[3].replace(b"2.22886148E+01", b"2.22886149E+01")
    with pytest.raises(ValueError, match="NASA fields changed: 4"):
        audit(RECORD, b"".join([*lines[:3], coefficient, *lines[4:]]))
    # The last record line carries 60 significant columns; its final coefficient must be compared too.
    last = lines[5].replace(b"2.92377865E+01", b"2.92377866E+01")
    assert last != lines[5]
    with pytest.raises(ValueError, match="NASA fields changed: 6"):
        audit(RECORD, b"".join([*lines[:5], last, *lines[6:]]))
    with pytest.raises(ValueError, match="NASA line count changed"):
        audit(RECORD, RECORD + b"\n")
    with pytest.raises(ValueError, match="vacuous NASA preservation audit"):
        audit(b"THERMO\nEND\n", b"THERMO\nEND\n")


def _fake_conversion(reproducer, tmp_path, *, ck2yaml_returncode=0, validation=None):
    calls = []
    validation = validation or {"cantera_version": "3.1.0", "species": 2, "reactions": 3, "warnings": ["w"]}

    def normalize(content, file):
        return SimpleNamespace(data=content, model_dump=lambda exclude: {"file": file, "bytes": len(content)})

    def run(cmd, **kwargs):
        calls.append(cmd)
        if "cantera.ck2yaml" in cmd:
            Path(cmd[cmd.index("--output") + 1]).write_text("phases: []\nreactions: []\n")
            return subprocess.CompletedProcess(cmd, ck2yaml_returncode, "converted", "")
        return subprocess.CompletedProcess(cmd, 0, json.dumps(validation), "")

    reproducer.patch(
        normalize_for_ck2yaml=normalize,
        audit_nasa_preservation=lambda raw, normalized: 1,
        subprocess=SimpleNamespace(run=run),
        parse_ck2yaml_log=lambda text: {"refusals": [], "text": text},
        trim_unsound_plog_levels=lambda model, pressures: (model, [{"removed_pressure_pa": 1.0}], []),
        cantera_validation_refusals=lambda warnings, sources, holdout_max_temperature_k: [],
        mechanism=lambda raw: (["A", "B"], [{}, {}]),
    )
    raw = b"REACTIONS\nA=B 1 0 0\n REV / 1.0 0 0 /\n REV / 0.0 0 0 /\nEND\n"
    pins = {"mechanism": {"filename": "m.inp", "unique_species": 2}, "thermo": {"filename": "t.dat"}}
    return calls, raw, pins


@pytest.mark.parametrize("key,reviews", [("v_old", slice(2, None)), ("v_new", slice(None, 2))])
def test_convert_records_the_audit_and_explains_the_reaction_count(reproducer, tmp_path, key, reviews):
    calls, raw, pins = _fake_conversion(reproducer, tmp_path)
    out = tmp_path / key
    audit = reproducer.g["convert"](key, raw, RECORD, pins, out, Path("python"), [101325.0], 1100.0)
    assert audit["source_reaction_entries"] == 2
    assert audit["nonzero_reverse_rates"] == 1
    assert audit["zero_reverse_rates"] == 1
    assert audit["converted_reactions"] == 3
    assert audit["plog_removed_pressures_pa"] == [1.0]
    expected = [dataclasses.asdict(r) for r in reproducer.g["REVIEWED_NASA_WARNINGS"][reviews]]
    assert audit["reviewed_warnings"] == expected
    assert audit["normalization"] == {
        "mechanism": {"file": "m.inp", "bytes": len(raw)},
        "thermo": {"file": "t.dat", "bytes": len(RECORD)},
    }
    assert (calls[0][2], calls[1][1]) == ("cantera.ck2yaml", "-c")
    for name in ("mechanism.normalized.inp", "thermo.normalized.inp", "trimmed.yaml", "ck2yaml.stdout.log"):
        assert (out / name).exists()
    stored = json.loads((out / "conversion-audit.json").read_text())
    assert stored["audit"] == json.loads(json.dumps(audit))


@pytest.mark.parametrize(
    "change,message",
    [
        ("ck2yaml", "v_old conversion"),
        ("plog", "unsound PLOG in scored domain"),
        ("validation", "Cantera validation"),
        ("warning", "unreviewed warning"),
        ("reactions", "unexplained converter reaction count"),
        ("species", "converted species changed"),
    ],
)
def test_convert_fails_closed(reproducer, tmp_path, change, message):
    validation = {"cantera_version": "3.1.0", "species": 2, "reactions": 3, "warnings": []}
    if change == "reactions":
        validation["reactions"] = 4
    elif change == "species":
        validation["species"] = 3
    _calls, raw, pins = _fake_conversion(
        reproducer, tmp_path, ck2yaml_returncode=int(change == "ck2yaml"), validation=validation
    )
    if change == "plog":
        reproducer.patch(trim_unsound_plog_levels=lambda model, pressures: (model, [], ["refused"]))
    elif change == "warning":
        reproducer.patch(cantera_validation_refusals=lambda *args, **kwargs: ["unreviewed"])
    elif change == "validation":
        run = reproducer.g["subprocess"].run

        def failing(cmd, **kwargs):
            result = run(cmd, **kwargs)
            return result if "cantera.ck2yaml" in cmd else subprocess.CompletedProcess(cmd, 1, "", "boom")

        reproducer.patch(subprocess=SimpleNamespace(run=failing))
    with pytest.raises(ValueError, match=message):
        reproducer.g["convert"]("v_old", raw, RECORD, pins, tmp_path / "out", Path("python"), [], 1100.0)


def _holdout_xml(doi, pressure_units="atm"):
    return (
        "<experiment><bibliographyLink><referenceDOI>doi.org/"
        f"{doi}</referenceDOI><description>Study {doi}</description></bibliographyLink>"
        f'<commonProperties><property name="pressure" units="{pressure_units}"><value>1.0</value></property>'
        "</commonProperties></experiment>"
    ).encode()


def _build_inputs(reproducer, tmp_path, monkeypatch, *, page_links=True, pressure_units="atm", bib_split=False):
    g = reproducer.g
    home = tmp_path / "home"
    evidence, run = home / "evidence", home / "run"
    for directory in (evidence, run, home / "runs/i127-holdout-sim/input", home / "runs/i117-contract"):
        directory.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    old = b"! LLNL iso-octane v3\n" * 8
    (evidence / "ic8-v3.txt").write_bytes(old)
    (evidence / "c5-2015.txt").write_bytes(b"new mechanism")
    (run / "prf_v3_therm_dat.txt").write_bytes(b"old thermo")
    (home / "runs/i127-holdout-sim/input/nuigmech_c5_july2015.therm").write_bytes(b"new thermo")
    (home / "runs/i117-contract/i117-glossary.pdf").write_bytes(b"glossary")
    base = "https://combustion.llnl.gov/sites/combustion/files/"
    companions = ["ic8_ver3_mech.txt", "prf_v3_therm_dat.txt"][: 2 if page_links else 1]
    (run / "llnl-iso-page-2026-10-02.html").write_text(" ".join(base + name for name in companions))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dev.xml", _holdout_xml("10.1/dev"))
        archive.writestr("dev2.xml", _holdout_xml("10.1/other" if bib_split else "10.1/dev"))
        archive.writestr("x00011009.xml", _holdout_xml("10.1/holdout", pressure_units))
    (evidence / "pentane_indirect_v2.5.zip").write_bytes(buffer.getvalue())

    def item(path):
        return {"path": path, "rows": 1, "observations": 1, "acceptance": "accepted", "refusal_reason": None}

    split = {
        "archive": {"filename": "pentane_indirect_v2.5.zip"},
        "studies": [
            {"set": "development", "files": [item("dev.xml"), item("dev2.xml")]},
            {"set": "holdout", "files": [item("x00011009.xml")]},
        ],
    }
    conversions = []
    pentane = dict(g["PENTANE"])
    pentane["frozen_split"] = lambda archive, fuel, holdout_doi: split
    reproducer.patch(
        EVIDENCE=evidence,
        RUN=run,
        PENTANE=pentane,
        source_pin=lambda filename, url, date, raw, digest: {
            "filename": filename,
            "source_url": url,
            "bytes": len(raw),
        },
        pinned=lambda path, pin: path.read_bytes(),
        mechanism=lambda raw: (["A"], [{}, {}]),
        thermo_inventory=lambda raw: ({"A": {"C": 1}}, 1),
        xml_temperature_max=lambda raw: 1100.0,
        convert=lambda key, *args: conversions.append((key, args)) or {"key": key},
        mapped_truth=lambda *args: {"old_species": [], "new_species": [], "ground_truth": []},
    )
    return conversions


def test_build_assembles_the_contract_from_pinned_local_bytes(reproducer, tmp_path, monkeypatch):
    conversions = _build_inputs(reproducer, tmp_path, monkeypatch)
    built = reproducer.g["build"](tmp_path / "out", Path("python"))
    assert built["case"] == "iso-pentane-low-t"
    assert built["mechanisms"]["v_old"]["unique_species"] == 1
    assert built["mechanisms"]["v_old"]["reaction_entries"] == 2
    assert built["thermo_sources"]["v_old"]["covered_mechanism_species"] == 1
    assert built["thermo_sources"]["v_new"]["nasa_records"] == 1
    assert built["holdout_max_temperature_k"] == 1100.0
    assert built["holdout_file_max_temperature_k"] == {"x00011009.xml": 1100.0}
    assert built["conversion"] == {"v_old": {"key": "v_old"}, "v_new": {"key": "v_new"}}
    assert [key for key, _args in conversions] == ["v_old", "v_new"]
    assert all(args[-2:] == ([101325.0], 1100.0) for _key, args in conversions)
    studies = built["split"]["studies"]
    assert [s["doi"] for s in studies] == ["10.1/dev", "10.1/holdout"]
    assert studies[0]["id"].startswith("Bugler") and studies[1]["id"].startswith("Zhong")
    assert len(built["identity"]["evidence"]) == 9


@pytest.mark.parametrize(
    "change,message",
    [
        ("page", "does not link both companions"),
        ("units", "unknown scored pressure units"),
        ("bibliography", "study bibliography mismatch"),
        ("coverage", "thermo does not cover every mechanism species"),
    ],
)
def test_build_fails_closed(reproducer, tmp_path, monkeypatch, change, message):
    _build_inputs(
        reproducer,
        tmp_path,
        monkeypatch,
        page_links=change != "page",
        pressure_units="bar" if change == "units" else "atm",
        bib_split=change == "bibliography",
    )
    if change == "coverage":
        reproducer.patch(thermo_inventory=lambda raw: ({}, 1))
    with pytest.raises(ValueError, match=message):
        reproducer.g["build"](tmp_path / "out", Path("python"))


def _main(reproducer, monkeypatch, tmp_path, built, *args):
    contract = tmp_path / "contract.json"
    contract.write_bytes(CONTRACT.read_bytes())
    reproducer.patch(CONTRACT=contract, build=lambda run_dir, python: built, load_split=lambda c, path: {"rows": 1})
    monkeypatch.setattr(sys, "argv", ["reproduce.py", "--run-dir", str(tmp_path / "run"), *args])
    reproducer.g["main"]()
    return contract


def test_main_checks_and_rewrites_the_frozen_contract(reproducer, monkeypatch, tmp_path, capsys):
    frozen = json.loads(CONTRACT.read_bytes())
    _main(reproducer, monkeypatch, tmp_path, frozen)
    output = capsys.readouterr().out
    assert output.count("structural absence verified") == len(frozen["ground_truth"])
    assert 'PASS split: {"rows": 1}' in output
    contract = _main(reproducer, monkeypatch, tmp_path, frozen, "--write-contract")
    assert contract.read_bytes() == CONTRACT.read_bytes()


def test_main_refuses_a_contract_that_differs_from_the_evidence(reproducer, monkeypatch, tmp_path):
    changed = json.loads(CONTRACT.read_bytes())
    changed["leakage_limits"] = [*changed["leakage_limits"], "one more limit"]
    with pytest.raises(ValueError, match="frozen contract differs from reproduced evidence"):
        _main(reproducer, monkeypatch, tmp_path, changed)
