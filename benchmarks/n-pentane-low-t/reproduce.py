"""Reproduce the frozen n-pentane contract without running kinetics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from carmel.benchmarks.chemkin import mechanism  # noqa: E402
from carmel.benchmarks.n_heptane import _read_pinned_root_file  # noqa: E402
from carmel.benchmarks.n_pentane import (  # noqa: E402
    Contract,
    formula_inventory,
    load_contract,
    load_split,
    require_structural_absence,
    series_counts,
)
from carmel.benchmarks.structure import (  # noqa: E402
    classify,
    features,
    pentane_low_temperature_species,
    reaction_signature,
)
from carmel.services.respecth import RespecthRefusal, parse_idt_record  # noqa: E402
from carmel.services.respecth_archive import PinnedArchive  # noqa: E402
from carmel.services.respecth_series import parse_series_record  # noqa: E402

EVIDENCE = Path("/home/alon/runs/i116-alkane-census/evidence")
OLD = EVIDENCE / "nc5-49.txt"
NEW = EVIDENCE / "c5-2015.txt"
ARCHIVE = EVIDENCE / "pentane_indirect_v2.5.zip"
CONTRACT = REPO / "benchmarks/n-pentane-low-t/contract.json"
DEVELOPMENT_DOI = "10.1016/j.combustflame.2015.09.014"
HOLDOUT_DOI = "10.1016/j.proci.2016.05.048"
ARCHIVE_FILENAME = "pentane_indirect_v2.5.zip"

RULES = {
    "G1": "One linear five-carbon OOQOOH radical becomes one linear five-carbon P(OOH)2 radical; elemental balance.",
    "G2": (
        "One linear five-carbon P(OOH)2 radical becomes one closed-shell hydroperoxy cyclic ether plus OH; "
        "elemental balance."
    ),
    "G3": (
        "One linear five-carbon P(OOH)2 radical becomes one closed-shell olefinic hydroperoxide plus HO2; "
        "elemental balance."
    ),
    "G4": (
        "One linear five-carbon OOQOOH radical directly becomes one closed-shell olefinic hydroperoxide plus HO2; "
        "elemental balance."
    ),
}
NAMES = {
    "G1": "alternative OOQOOH H-shift to P(OOH)2",
    "G2": "P(OOH)2 to hydroperoxy cyclic ether + OH",
    "G3": "P(OOH)2 to olefinic hydroperoxide + HO2",
    "G4": "direct OOQOOH concerted HO2 elimination",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pin(raw: bytes, *, filename: str, source_url: str, retrieval_date: str) -> dict[str, object]:
    labels, reactions = mechanism(raw)
    return {
        "filename": filename,
        "source_url": source_url,
        "retrieval_date": retrieval_date,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "unique_species": len(labels),
        "reaction_entries": len(reactions),
    }


def species_record(label: str, formula: dict[str, int], *, required: bool = True) -> dict[str, object]:
    smiles = pentane_low_temperature_species(label)
    if smiles is None:
        require(not required, f"UNRESOLVED member species: {label}")
        return {
            "label": label,
            "status": "UNRESOLVED",
            "smiles": None,
            "formula": formula,
            "source": "no confident structure in the tested low-temperature label grammar",
        }
    require(dict(features(smiles).atoms) == formula, f"structure formula conflicts with mechanism composition: {label}")
    return {
        "label": label,
        "status": "RESOLVED",
        "smiles": smiles,
        "formula": dict(features(smiles).atoms),
        "source": (
            "tested Bugler/NUIG straight-chain C5 label grammar"
            if label.startswith("C5")
            else "tested N?CnKETij grammar"
            if "KET" in label
            else "explicit small-radical structure"
        ),
    }


def mapped_ground_truth(
    old_source: bytes, new_source: bytes
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    old_labels, old_reactions = mechanism(old_source)
    _, new_reactions = mechanism(new_source)
    old_formulas = formula_inventory(old_labels, old_reactions)
    old_structures = {
        label: pentane_low_temperature_species(label)
        for label in old_labels
        if pentane_low_temperature_species(label) is not None
    }
    old_signatures = {
        reaction_signature(
            tuple(old_structures[label] for label in reaction["reactants"]),
            tuple(old_structures[label] for label in reaction["products"]),
        )
        for reaction in old_reactions
        if all(label in old_structures for label in (*reaction["reactants"], *reaction["products"]))
    }
    members: dict[str, list[dict[str, object]]] = {key: [] for key in RULES}
    labels: set[str] = set()
    for reaction in new_reactions:
        names = (*reaction["reactants"], *reaction["products"])
        structures = {name: pentane_low_temperature_species(name) for name in names}
        if not all(structures.values()):
            continue
        left = tuple(structures[name] for name in reaction["reactants"])
        right = tuple(structures[name] for name in reaction["products"])
        classes = classify(left, right, carbon_count=5) & RULES.keys()
        if not classes:
            continue
        signature = reaction_signature(left, right)
        require(signature not in old_signatures, f"baseline reaction signature already present: {reaction['equation']}")
        candidate_formulas = {name: dict(features(structures[name]).atoms) for name in names}
        require_structural_absence(
            reaction,
            candidate_formulas,
            signature,
            old_reactions,
            old_formulas,
            old_structures,
        )
        labels.update(names)
        for class_id in classes:
            members[class_id].append(
                {
                    **reaction,
                    "absence": "structural-signature-and-exhaustive-formula-screen",
                }
            )
    expected = {"G1": 30, "G2": 30, "G3": 12, "G4": 12}
    require({key: len(value) for key, value in members.items()} == expected, "member counts changed")
    new_species = []
    for label in sorted(labels):
        smiles = pentane_low_temperature_species(label)
        assert smiles is not None
        new_species.append(species_record(label, dict(features(smiles).atoms)))
    old_species = [species_record(label, old_formulas[label], required=False) for label in sorted(old_labels)]
    ground_truth = []
    for class_id in RULES:
        class_members = members[class_id]
        class_labels = {label for item in class_members for label in (*item["reactants"], *item["products"])}
        ground_truth.append(
            {
                "id": class_id,
                "name": NAMES[class_id],
                "scope": "primary",
                "rule": RULES[class_id],
                "reactant_formula": {"C": 5, "H": 11, "O": 4},
                "member_count": len(class_members),
                "resolved_count": len(class_members),
                "member_species_count": len(class_labels),
                "resolved_species_count": len(class_labels),
                "members": class_members,
            }
        )
    return new_species, old_species, ground_truth


def _study_doi(root: ElementTree.Element) -> str:
    return (root.findtext("bibliographyLink/referenceDOI") or "").removeprefix("doi.org/")


def _has_n_pentane(root: ElementTree.Element) -> bool:
    return any(
        link.get("preferredKey") == "NC5H12" for link in root.findall("commonProperties/property/component/speciesLink")
    )


def _response_counts(root: ElementTree.Element) -> tuple[int, int]:
    rows = observations = 0
    for group in root.findall("dataGroup"):
        response_ids = {
            prop.get("id")
            for prop in group.findall("property")
            if prop.get("name") in {"ignition delay", "composition"}
        }
        if not response_ids:
            continue
        points = group.findall("dataPoint")
        rows += len(points)
        observations += sum(sum(child.tag in response_ids for child in point) for point in points)
    return rows, observations


def frozen_split(archive_bytes: bytes) -> dict[str, object]:
    archive_pin = {
        "filename": ARCHIVE_FILENAME,
        "source_url": "https://files.osf.io/v1/resources/nbmzv/providers/osfstorage/671690d83dea2353a7b93476?version=1",
        "retrieval_date": "2026-10-02",
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "bytes": len(archive_bytes),
        "unique_species": None,
        "reaction_entries": None,
        "osf_path": "/ReactionKinetics/indirect/pentane/pentane_indirect_v2.5.zip",
        "osf_file_id": "671690d83dea2353a7b93476",
        "osf_version": 1,
    }
    parser_archive = PinnedArchive(
        name=ARCHIVE_FILENAME,
        osf_path=str(archive_pin["osf_path"]),
        osf_file_id=str(archive_pin["osf_file_id"]),
        osf_version=1,
        size=len(archive_bytes),
        sha256=str(archive_pin["sha256"]),
    )
    development: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        for name in sorted(source.namelist()):
            if not name.endswith(".xml"):
                continue
            raw = source.read(name)
            root = ElementTree.fromstring(raw)
            doi = _study_doi(root)
            if doi not in {DEVELOPMENT_DOI, HOLDOUT_DOI} or not _has_n_pentane(root):
                continue
            rows, observations = _response_counts(root)
            target = development if doi == DEVELOPMENT_DOI else holdout
            parser = parse_idt_record if doi == DEVELOPMENT_DOI else parse_series_record
            try:
                record = parser(raw, parser_archive, name)
                acceptance = "supported"
                refusal_reason = None
                _, _, low_rows, low_observations = series_counts(record)
            except RespecthRefusal as exc:
                acceptance = "refused"
                refusal_reason = exc.reason.value
                low_rows = low_observations = 0
            target.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "rows": rows,
                    "observations": observations,
                    "low_temperature_rows": low_rows,
                    "low_temperature_observations": low_observations,
                    "acceptance": acceptance,
                    "refusal_reason": refusal_reason,
                }
            )
    return {
        "archive": archive_pin,
        "studies": [
            {
                "id": "Bugler et al. 2016 n-pentane ignition delay",
                "bibliography": "J. Bugler et al., Combustion and Flame 163 (2016) 138-156",
                "doi": "10.1016/j.combustflame.2015.09.014",
                "set": "development",
                "files": development,
            },
            {
                "id": "Bugler et al. 2017 n-pentane jet-stirred reactors",
                "bibliography": "J. Bugler et al., Proceedings of the Combustion Institute 36 (2017) 441-448",
                "doi": "10.1016/j.proci.2016.05.048",
                "set": "holdout",
                "files": holdout,
            },
        ],
    }


def build_contract(old_source: bytes, new_source: bytes, archive_bytes: bytes) -> dict[str, object]:
    new_species, old_species, ground_truth = mapped_ground_truth(old_source, new_source)
    split = frozen_split(archive_bytes)
    return {
        "schema_version": 1,
        "case": "n-pentane-low-t",
        "mechanisms": {
            "v_old": pin(
                old_source,
                filename="nc5_49_mech.dat",
                source_url="https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nc5_49_mech.dat",
                retrieval_date="2026-10-14",
            ),
            "v_new": pin(
                new_source,
                filename="nuigmech_c5_july2015_1215.chem",
                source_url="https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nuigmech_c5_july2015_1215.chem",
                retrieval_date="2026-10-14",
            ),
        },
        "support_sources": [split["archive"]],
        "identity": {
            "status": "UNVERIFIED",
            "historical_identity_caveat": (
                "Bugler names Healy as the starting mechanism, but the current nc5_49 and July-2015 files are not "
                "authenticated publication-era snapshots."
            ),
            "attempts": [
                "Inspected both mechanism headers and filenames for a publication date or immutable release "
                "identifier.",
                "Read Bugler 2015 section 2, which names Healy's model as the starting mechanism but does not "
                "fingerprint nc5_49 bytes.",
                "Compared the retrieved endpoint species and reaction-entry counts with the census evidence.",
            ],
            "evidence": [
                "Bugler et al. 2015 section 2 states that the model is based on the n-pentane mechanism developed "
                "by Healy et al.",
                "The nc5_49 filename and 293/1593 inventory are consistent with the retrieved historical release.",
                "No primary source located ties either frozen SHA-256 to the exact submitted/published mechanism "
                "bytes.",
            ],
        },
        "new_species": new_species,
        "old_species": old_species,
        "ground_truth": ground_truth,
        "split": split,
        "scoring": {
            "primary_unit": "reaction-type-once",
            "primary_types": ["G1", "G2", "G3", "G4"],
            "recovered": (
                "A structurally accepted type earns one unit once; duplicate entries, fitted rates, and extra members "
                "earn no additional credit."
            ),
            "baseline_present": (
                "Classify the generated baseline first; a type already present receives baseline coverage and zero "
                "revision credit."
            ),
            "secondary_metric": (
                "For each measured species, mean absolute log10(max(x_sim,L_s)/max(x_exp,L_s)) over its frozen "
                "rows; the overall score is the unweighted mean of species scores."
            ),
            "species_aggregation": (
                "Rows are equally weighted within species and species are equally weighted; report every "
                "per-species score and scored-row count."
            ),
            "zero_and_detection_limit_policy": (
                "L_s is the stated detection limit where supplied; otherwise half the smallest strictly positive "
                "frozen observation for that species. Exact zeros and values below L_s are clamped to L_s. A species "
                "with no positive value and no stated limit is refused, never assigned zero error."
            ),
            "low_temperature_subset": "600-900 K inclusive; reported separately",
            "requires": "jsr-adapter",
            "refusals": [
                "missing or nonfinite simulated mole fraction",
                "negative simulated or experimental mole fraction",
                "species with no stated detection limit and no positive frozen observation",
                "unsupported reactor condition or solver failure",
            ],
            "locking": (
                "Lock candidate, baseline, contract hash, structural matcher, JSR adapter and simulator before one "
                "holdout evaluation; no holdout-driven retries, tuning or exclusions."
            ),
        },
        "leakage_limits": [
            "The Bugler ignition-delay study validated v_new and is development data, never holdout.",
            "The JSR holdout shares mechanism-community authors with the revision; historical blindness is not "
            "claimed.",
            "Keep v_new bytes, mapped reactions and labels out of the revision agent and retrieval tools until "
            "candidate locking.",
            "Thermochemistry and rate-rule changes made with the revision are excluded from structural ground truth.",
            "Korcek chemistry was not included in v_new and is not a scored type.",
            "ChemKED n-pentane records are NOx-doped and excluded from both sets.",
            "Hash drift, split-count drift and unresolved eligible old structures fail the benchmark rather than "
            "shrinking it.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-contract", action="store_true", help="write the reproduced JSON instead of auditing")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--v-old", type=Path, default=OLD)
    parser.add_argument("--v-new", type=Path, default=NEW)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    frozen = None if args.write_contract else load_contract(args.contract)
    old_source = _read_pinned_root_file(args.v_old)
    if frozen is not None:
        frozen.mechanisms["v_old"].verify(old_source)
    new_source = _read_pinned_root_file(args.v_new)
    if frozen is not None:
        frozen.mechanisms["v_new"].verify(new_source)
    archive_bytes = _read_pinned_root_file(args.archive)
    if frozen is not None:
        frozen.split.archive.verify(archive_bytes)
    reproduced = build_contract(old_source, new_source, archive_bytes)
    validated = Contract.model_validate(reproduced)
    if args.write_contract:
        args.contract.parent.mkdir(parents=True, exist_ok=True)
        args.contract.write_text(json.dumps(validated.model_dump(mode="json"), indent=2) + "\n")
    else:
        assert frozen is not None
        require(frozen.model_dump(mode="json") == validated.model_dump(mode="json"), "frozen contract changed")
        totals = load_split(frozen, archive_bytes)
        for truth in frozen.ground_truth:
            print(
                f"PASS {truth.id}: {truth.member_count} members; {truth.resolved_count} resolved; "
                f"{truth.resolved_species_count}/{truth.member_species_count} species; structural absence verified"
            )
        print("PASS split:", json.dumps(totals, sort_keys=True))
        print("PASS mechanisms: 293/1593 old; 697/3214 new")
        print("IDENTITY: UNVERIFIED")


if __name__ == "__main__":
    main()
