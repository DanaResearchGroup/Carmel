"""Reproduce case 3 from pinned local bytes; conversion only, never simulation."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import re
import runpy
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from carmel.benchmarks.cantera_worker import (  # noqa: E402
    REVIEWED_NASA_WARNINGS,
    cantera_validation_refusals,
    parse_ck2yaml_log,
    trim_unsound_plog_levels,
)
from carmel.benchmarks.chemkin import mechanism, normalize_for_ck2yaml  # noqa: E402
from carmel.benchmarks.iso_pentane import Contract, load_split, xml_temperature_max  # noqa: E402
from carmel.benchmarks.n_pentane import formula_key, require_structural_absence  # noqa: E402
from carmel.benchmarks.structure import classify, features, reaction_signature  # noqa: E402

EVIDENCE = Path.home() / "runs/i116-alkane-census/evidence"
RUN = Path.home() / "runs/i137-ic5-contract"
CONTRACT = REPO / "benchmarks/iso-pentane-low-t/contract.json"
HEPTANE = runpy.run_path(str(REPO / "benchmarks/n-heptane-low-t/reproduce.py"))
PENTANE = runpy.run_path(str(REPO / "benchmarks/n-pentane-low-t/reproduce.py"))
CASE2 = json.loads((REPO / "benchmarks/n-pentane-low-t/contract.json").read_bytes())
CASE1 = json.loads((REPO / "benchmarks/n-heptane-low-t/contract.json").read_bytes())
THERMO_URL = "https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/"


def require(condition, message):
    if not condition:
        raise ValueError("NEEDS-INPUT: " + message)


def pinned(path, pin):
    raw = path.read_bytes()
    require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], f"PIN CHANGED: {path}")
    return raw


def source_pin(filename, url, date, raw, digest):
    require(hashlib.sha256(raw).hexdigest() == digest, f"PIN CHANGED: {filename}")
    return dict(
        filename=filename,
        source_url=url,
        retrieval_date=date,
        sha256=digest,
        bytes=len(raw),
        unique_species=None,
        reaction_entries=None,
    )


def thermo_inventory(raw):
    # Reuse case 1's fixed-column formula parser, but reject conflicting duplicates
    # before accepting its inventory (the converter uses the first record).
    lines = raw.splitlines(keepends=True)
    formulas = {}
    count = 0
    for i, line in enumerate(lines):
        if line[79:80] != b"1":
            continue
        item = HEPTANE["thermo"](b"".join(lines[i : i + 4]))
        for label, formula in item.items():
            require(label not in formulas or formulas[label] == formula, f"conflicting formula: {label}")
            formulas.setdefault(label, formula)
            count += 1
    require(count > 0, "empty thermo inventory")
    return formulas, count


def audit_nasa_preservation(raw, normalized):
    before, after = raw.splitlines(), normalized.splitlines()
    require(len(before) == len(after), "NASA line count changed")
    count = 0
    for i, line in enumerate(before):
        if line[79:80] != b"1":
            continue
        count += 1
        for offset, width in ((0, 75), (1, 75), (2, 75), (3, 60)):
            require(before[i + offset][:width] == after[i + offset][:width], f"NASA fields changed: {i + offset + 1}")
    require(count > 0, "vacuous NASA preservation audit")
    return count


def convert(key, raw, thermo, pins, out, python, pressures, max_temperature):
    out.mkdir(parents=True, exist_ok=True)
    norm = {}
    sources = []
    for kind, content in (("mechanism", raw), ("thermo", thermo)):
        item = normalize_for_ck2yaml(content, file=pins[kind]["filename"])
        (out / f"{kind}.normalized.inp").write_bytes(item.data)
        norm[kind] = item.model_dump(exclude={"data"})
        sources.append(item.data)
    print(key, "NASA preservation:", audit_nasa_preservation(thermo, sources[1]), "records")
    cmd = [
        str(python),
        "-m",
        "cantera.ck2yaml",
        "--permissive",
        "--verbose",
        "--no-validate",
        "--input",
        str(out / "mechanism.normalized.inp"),
        "--thermo",
        str(out / "thermo.normalized.inp"),
        "--output",
        str(out / "converted.yaml"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (out / "ck2yaml.stdout.log").write_text(result.stdout)
    (out / "ck2yaml.stderr.log").write_text(result.stderr)
    log = parse_ck2yaml_log(result.stdout + result.stderr)
    require(result.returncode == 0 and not log["refusals"], f"{key} conversion: {log['refusals']}")
    model = yaml.safe_load((out / "converted.yaml").read_text())
    trimmed, trims, refusals = trim_unsound_plog_levels(model, pressures)
    require(not refusals, f"{key} unsound PLOG in scored domain: {refusals}")
    (out / "trimmed.yaml").write_text(yaml.safe_dump(trimmed, sort_keys=False))
    code = """import cantera as ct, json, sys, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    gas = ct.Solution(sys.argv[1])
print(json.dumps(dict(cantera_version=ct.__version__, species=gas.n_species,
                     reactions=gas.n_reactions, warnings=[str(w.message) for w in caught])))
"""
    result = subprocess.run(
        [str(python), "-c", code, str(out / "trimmed.yaml")], capture_output=True, text=True, check=False
    )
    (out / "validation.stdout.log").write_text(result.stdout)
    (out / "validation.stderr.log").write_text(result.stderr)
    require(result.returncode == 0, f"{key} Cantera validation: {result.stderr}")
    validation = json.loads(result.stdout)
    require(
        not cantera_validation_refusals(
            validation["warnings"], tuple(sources), holdout_max_temperature_k=max_temperature
        ),
        f"{key} unreviewed warning: {validation['warnings']}",
    )
    rev = re.findall(rb"^\s*REV\s*/([^/]*)/", raw, re.M | re.I)
    nonzero = sum(float(r.split()[0]) != 0 for r in rev)
    _, entries = mechanism(raw)
    require(validation["reactions"] == len(entries) + nonzero, f"{key} unexplained converter reaction count")
    require(validation["species"] == pins["mechanism"]["unique_species"], f"{key} converted species changed")
    reviews = REVIEWED_NASA_WARNINGS[2:] if key == "v_old" else REVIEWED_NASA_WARNINGS[:2]
    audit = dict(
        cantera_version=validation["cantera_version"],
        source_reaction_entries=len(entries),
        nonzero_reverse_rates=nonzero,
        zero_reverse_rates=len(rev) - nonzero,
        converted_reactions=validation["reactions"],
        warnings=validation["warnings"],
        reviewed_warnings=[dataclasses.asdict(r) for r in reviews],
        plog_removed_pressures_pa=[r["removed_pressure_pa"] for r in trims],
        normalization=norm,
    )
    (out / "conversion-audit.json").write_text(json.dumps(dict(audit=audit, converter=log, plog_trims=trims), indent=2))
    print(key, "conversion PASS:", len(entries), "+", nonzero, "REV =", validation["reactions"])
    return audit


def mapped_truth(old, new, old_thermo, new_thermo, glossary):
    dictionary = HEPTANE["glossary"](glossary)
    old_labels, old_reactions = mechanism(old)
    new_labels, new_reactions = mechanism(new)
    old_formulas, _ = thermo_inventory(old_thermo)
    new_formulas, _ = thermo_inventory(new_thermo)
    old_species = HEPTANE["inventory"](old_labels, dictionary, old_formulas)
    old_structures = {s["label"]: s["smiles"] for s in old_species if s["smiles"]}
    # Exhaustively audit possible source formulas before the structural scan, so
    # an unresolved source cannot silently disappear from the member denominator.
    for label in new_labels:
        if new_formulas[label] == {"C": 5, "H": 11, "O": 4}:
            smiles = dictionary.get(label, {}).get("smiles")
            require(smiles is not None, f"UNRESOLVED C5 source: {label}")
            require(Counter(features(smiles).atoms) == Counter(new_formulas[label]), f"formula conflict: {label}")
    members = {g: [] for g in PENTANE["RULES"]}
    # Necessary formula profiles exclude fragmentation, which is not scored here.
    # Classification itself remains the job of the shared molecular graph rules.
    target_profiles = {
        tuple(sorted(tuple(sorted(f.items())) for f in formulas))
        for formulas in (
            ({"C": 5, "H": 11, "O": 4},),
            ({"C": 5, "H": 10, "O": 3}, {"H": 1, "O": 1}),
            ({"C": 5, "H": 10, "O": 2}, {"H": 1, "O": 2}),
        )
    }
    for reaction in new_reactions:
        sides = (reaction["reactants"], reaction["products"])
        eligible = any(
            len(source) == 1
            and new_formulas[source[0]] == {"C": 5, "H": 11, "O": 4}
            and formula_key(target, new_formulas) in target_profiles
            for source, target in (sides, sides[::-1])
        )
        if not eligible:
            continue
        names = (*sides[0], *sides[1])
        structures = {name: dictionary.get(name, {}).get("smiles") for name in names}
        require(all(structures.values()), f"UNRESOLVED source-family participant: {reaction['equation']}")
        for name in names:
            require(
                Counter(features(structures[name]).atoms) == Counter(new_formulas[name]), f"formula conflict: {name}"
            )
        left, right = (tuple(structures[name] for name in side) for side in sides)
        groups = classify(left, right, carbon_count=5, skeleton="iso-pentane")
        if not groups:
            continue
        require_structural_absence(
            reaction, new_formulas, reaction_signature(left, right), old_reactions, old_formulas, old_structures
        )
        for group in groups:
            members[group].append(dict(reaction, absence="structural-signature-and-exhaustive-formula-screen"))
    ground_truth = []
    all_labels = set()
    for group, entries in members.items():
        require(bool(entries), f"zero members: {group}")
        names = {n for r in entries for n in (*r["reactants"], *r["products"])}
        all_labels.update(names)
        ground_truth.append(
            dict(
                id=group,
                name=PENTANE["NAMES"][group],
                scope="primary",
                rule=PENTANE["RULES"][group].replace("linear five-carbon", "2-methylbutane-skeleton"),
                reactant_formula={"C": 5, "H": 11, "O": 4},
                member_count=len(entries),
                resolved_count=len(entries),
                member_species_count=len(names),
                resolved_species_count=len(names),
                members=entries,
            )
        )
    new_species = HEPTANE["inventory"](sorted(all_labels), dictionary, new_formulas)
    return dict(old_species=old_species, new_species=new_species, ground_truth=ground_truth)


def build(run_dir, cantera_python):
    old = (EVIDENCE / "ic8-v3.txt").read_bytes()
    old_pin = source_pin(
        "ic8_ver3_mech.txt",
        "https://combustion.llnl.gov/sites/combustion/files/ic8_ver3_mech.txt",
        "2026-10-02",
        old,
        "2fc57d65d8dd7b4e9793f09639f41aca114aa878624f4b93b726ba2f39a68d5f",
    )
    labels, entries = mechanism(old)
    old_pin.update(unique_species=len(labels), reaction_entries=len(entries))
    new = pinned(EVIDENCE / "c5-2015.txt", CASE2["mechanisms"]["v_new"])
    old_thermo = (RUN / "prf_v3_therm_dat.txt").read_bytes()
    old_thermo_pin = source_pin(
        "prf_v3_therm_dat.txt",
        "https://combustion.llnl.gov/sites/combustion/files/prf_v3_therm_dat.txt",
        "2026-11-03",
        old_thermo,
        "acb98b7574d9568714b928444ebea8aae86e794168d1d597297dc2c00ff5ae4e",
    )
    new_thermo = (Path.home() / "runs/i127-holdout-sim/input/nuigmech_c5_july2015.therm").read_bytes()
    new_thermo_pin = source_pin(
        "nuigmech_c5_july2015.therm",
        THERMO_URL + "nuigmech_c5_july2015.therm",
        "2026-10-17",
        new_thermo,
        "c8e465ea61c061f466352b2e6a84c2d6854f255d33e7e1363aa4cd0a8d3ea9fa",
    )
    for pin, raw, mech in ((old_thermo_pin, old_thermo, old), (new_thermo_pin, new_thermo, new)):
        formulas, records = thermo_inventory(raw)
        species, _ = mechanism(mech)
        require(not set(species) - formulas.keys(), "thermo does not cover every mechanism species")
        pin.update(
            unique_species=len(formulas),
            reaction_entries=0,
            nasa_records=records,
            covered_mechanism_species=len(species),
        )
    page = (RUN / "llnl-iso-page-2026-10-02.html").read_bytes()
    page_pin = source_pin(
        "llnl-iso-page-2026-10-02.html",
        "https://combustion.llnl.gov/mechanisms/alkanes/iso-octane-version-3",
        "2026-10-02",
        page,
        "15c3fe15836b7a1f55460bc6f1034be54317ee4f47310054b224be26899aeb16",
    )
    require(
        old_pin["source_url"].encode() in page and old_thermo_pin["source_url"].encode() in page,
        "LLNL page does not link both companions",
    )
    glossary = pinned(Path.home() / "runs/i117-contract/i117-glossary.pdf", CASE1["support_sources"][0])
    archive = pinned(EVIDENCE / "pentane_indirect_v2.5.zip", CASE2["split"]["archive"])
    split = PENTANE["frozen_split"](archive, fuel="IC5H12", holdout_doi="10.1016/j.combustflame.2021.111678")
    bounds = {}
    pressures = []
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for study in split["studies"]:
            bibs = []
            for item in study["files"]:
                raw = z.read(item["path"])
                xml = ElementTree.fromstring(raw)
                bibs.append(
                    (
                        xml.findtext("bibliographyLink/referenceDOI").removeprefix("doi.org/"),
                        xml.findtext("bibliographyLink/description"),
                    )
                )
                if study["set"] == "holdout":
                    bounds[item["path"]] = xml_temperature_max(raw)
                    props = [p for p in xml.findall("commonProperties/property") if p.get("name") == "pressure"]
                    require(len(props) == 1 and props[0].get("units") == "atm", "unknown scored pressure units")
                    pressures.append(float(props[0].findtext("value")) * 101325)
                print(item["path"], item["rows"], item["observations"], item["acceptance"], item["refusal_reason"])
            require(len(set(bibs)) == 1, "study bibliography mismatch")
            study["doi"], study["bibliography"] = bibs[0]
            study["id"] = (
                "Bugler et al. 2016 iso-pentane ignition delay"
                if study["set"] == "development"
                else "Zhong et al. 2022 iso-pentane jet-stirred reactor"
            )
    conversions = {}
    for key, raw, thermo, mp, tp in (
        ("v_old", old, old_thermo, old_pin, old_thermo_pin),
        ("v_new", new, new_thermo, CASE2["mechanisms"]["v_new"], new_thermo_pin),
    ):
        conversions[key] = convert(
            key,
            raw,
            thermo,
            dict(mechanism=mp, thermo=tp),
            run_dir / key,
            cantera_python,
            pressures,
            max(bounds.values()),
        )
    truth = mapped_truth(old, new, old_thermo, new_thermo, glossary)
    return dict(
        schema_version=1,
        case="iso-pentane-low-t",
        mechanisms=dict(v_old=old_pin, v_new=CASE2["mechanisms"]["v_new"]),
        thermo_sources=dict(v_old=old_thermo_pin, v_new=new_thermo_pin),
        support_sources=[split["archive"]],
        structure_source=CASE1["support_sources"][0],
        companion_page=page_pin,
        companion_page_evidence="LLNL Iso-Octane Version 3 page lines 760-761 link prf_v3_therm_dat.txt "
        "as Thermodynamic parameters and ic8_ver3_mech.txt as Chemical kinetic mechanism.",
        conversion=conversions,
        holdout_max_temperature_k=max(bounds.values()),
        holdout_file_max_temperature_k=bounds,
        identity=dict(
            status="UNVERIFIED",
            historical_identity_caveat=(
                "A sound documented pre-2015 IC5 baseline, but exact publication-era bytes and direct "
                "parentage of NUIG C5 are not authenticated. v_new is the maintained case-2 artifact."
            ),
            attempts=[
                "Read exact mechanism headers, linked LLNL page, both thermo inventories, "
                "and Bugler2015 model-development sections; no immutable release was found."
            ],
            evidence=[
                *old.decode().splitlines()[:8],
                "LLNL release page cites Mehl et al. 2009; IC5H12 and its low-T submechanism "
                "are explicit. PRF thermo covers 874/874 mechanism species.",
            ],
        ),
        **truth,
        split=split,
        # Case 2's frozen scoring before its revision-2 target chemistry; case 3 adopts that revision separately.
        scoring={key: value for key, value in CASE2["scoring"].items() if key != "target_chemistry"},
        leakage_limits=[
            "v_new and the four Bugler2015 chemistry classes are shared with case 2; this tests transfer "
            "to a branched fuel, not an independent historical chemistry discovery.",
            "The development study shares authors, archive and mechanism validation with case 2. "
            "The fuel-specific files are distinct; study identity is shared across cases.",
            "The XML holdout bibliography is Zhong et al., Combustion and Flame 235 (2022) 111678, "
            "DOI 10.1016/j.combustflame.2021.111678. These authors differ from Bugler et al. and the study "
            "postdates the named 2015 revision, strengthening historical blindness; the maintained v_new "
            "bytes are not an authenticated 2015 snapshot, so absolute blindness is not established.",
            "The shared ReSpecTh archive is a common curation channel, not independent evidence of chemistry.",
            "Keep v_new, author glossary, mapped truth and labels out of revision prompts and retrieval "
            "until locking; score type recovery once with baseline credit, not rate or thermo tuning.",
            "Korcek chemistry is excluded. Preserve all reader refusals, including all 121 development "
            "IDT points; this contract currently provides no reader-supported development observations.",
            "Holdout includes pyrolysis x00011012 and oxidation x00011009-011 as one whole study. "
            "Its digitized numeric XML observations retain their provenance; no point is redigitized.",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-contract", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=RUN / "reproduction")
    parser.add_argument("--cantera-python", type=Path, default=Path.home() / "anaconda3/envs/t3_env/bin/python")
    args = parser.parse_args()
    reproduced = Contract.model_validate(build(args.run_dir, args.cantera_python))
    if args.write_contract:
        CONTRACT.write_text(reproduced.model_dump_json(indent=2) + "\n")
    else:
        frozen = Contract.model_validate_json(CONTRACT.read_bytes())
        require(frozen == reproduced, "frozen contract differs from reproduced evidence")
    totals = load_split(reproduced, EVIDENCE / "pentane_indirect_v2.5.zip")
    for truth in reproduced.ground_truth:
        print(
            f"PASS {truth.id}: {truth.member_count} members; {truth.resolved_count} resolved; "
            f"{truth.resolved_species_count}/{truth.member_species_count} species; structural absence verified"
        )
    print("PASS split:", json.dumps(totals, sort_keys=True))
    print("PASS thermo coverage: 874/874 old; 697/697 new. IDENTITY: UNVERIFIED")


if __name__ == "__main__":
    main()
