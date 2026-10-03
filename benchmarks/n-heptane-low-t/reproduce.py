"""Reproduce the frozen n-heptane contract without running kinetics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from carmel.benchmarks.n_heptane import (  # noqa: E402
    _MAX_FILE_BYTES,
    GroundTruth,
    Pin,
    _read_pinned_root_file,
    expected_reactant_formula,
    load_contract,
    load_split,
)


class PinnedInputError(ValueError):
    """A pinned input cannot be safely acquired or used."""


class ContractMismatch(ValueError):
    """A reproduced result differs from the frozen contract."""


class PinnedSnapshot(NamedTuple):
    path: Path
    data: bytes


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractMismatch(message)


def _source_bytes(source: Path | bytes) -> bytes:
    return source if isinstance(source, bytes) else _read_pinned_root_file(source)


def mechanism(source: Path | bytes) -> tuple[list[str], list[dict[str, object]]]:
    display = str(source) if isinstance(source, Path) else "<verified snapshot>"
    mode = None
    species: list[str] = []
    reactions: list[dict[str, object]] = []
    for number, raw in enumerate(_source_bytes(source).decode(errors="replace").splitlines(), 1):
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        head = line.split()[0].upper()
        if head in ("SPECIES", "SPEC"):
            mode = "species"
            species.extend(line.split()[1:])
            continue
        if head in ("REACTIONS", "REAC"):
            mode = "reactions"
            continue
        if head == "END":
            mode = None
            continue
        if mode == "species":
            species.extend(line.split())
        elif mode == "reactions" and "=" in line:
            match = re.match(r"(.+?)\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\s+", line)
            if not match:
                raise ValueError(f"unparsed reaction: {display}:{number}")
            equation = re.sub(r"\s+", "", match[1])
            sides = []
            for side in re.split("<=>|=>|=", equation):
                labels: list[str] = []
                for token in side.replace("(+M)", "").split("+"):
                    if token == "M":
                        continue
                    coefficient = re.fullmatch(r"(\d+)?(.+)", token)
                    if coefficient is None:
                        raise ContractMismatch(f"unparsed reaction coefficient: {display}:{number}")
                    labels.extend([coefficient[2]] * int(coefficient[1] or 1))
                sides.append(tuple(labels))
            reactions.append(dict(equation=equation, line=number, reactants=sides[0], products=sides[1]))
    return sorted(set(species)), reactions


def fetch_url(url: str, *, max_bytes: int = _MAX_FILE_BYTES) -> bytes:
    with urlopen(url, timeout=120) as response:  # noqa: S310 -- the URL is pinned in contract.json
        return response.read(max_bytes)


def _target_for(pin: Pin, directory: Path) -> Path:
    filename = pin.filename
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."} or ".." in filename:
        raise PinnedInputError(f"pinned filename must be a plain basename: {filename!r}")
    root = directory.resolve()
    target = root / filename
    try:
        target.resolve().relative_to(root)
    except ValueError as error:
        raise PinnedInputError(f"pinned target is not contained in download directory: {target}") from error
    return target


def _read_pinned(pin: Pin, target: Path) -> bytes:
    try:
        size = target.stat().st_size
    except OSError as error:
        raise PinnedInputError(f"cannot stat pinned input {target}") from error
    if size > pin.bytes:
        raise PinnedInputError(f"cached {pin.filename} exceeds frozen byte count")
    try:
        with target.open("rb") as stream:
            data = stream.read(pin.bytes + 1)
    except OSError as error:
        raise PinnedInputError(f"cannot read pinned input {target}") from error
    if len(data) > pin.bytes:
        raise PinnedInputError(f"cached {pin.filename} exceeds frozen byte count")
    pin.verify(data)
    return data


def _acquire_pinned_snapshot(
    pin: Pin, directory: Path, *, fetcher: Callable[[str], bytes] = fetch_url
) -> PinnedSnapshot:
    target = _target_for(pin, directory)
    if target.is_file():
        return PinnedSnapshot(target, _read_pinned(pin, target))
    data = fetcher(pin.source_url, max_bytes=pin.bytes + 1) if fetcher is fetch_url else fetcher(pin.source_url)
    if len(data) > pin.bytes:
        raise PinnedInputError(f"downloaded {pin.filename} exceeds frozen byte count")
    pin.verify(data)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
    except FileExistsError as error:
        raise PinnedInputError(f"pinned target appeared after validation: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except OSError as error:
        target.unlink(missing_ok=True)
        raise PinnedInputError(f"cannot publish pinned input {target}") from error
    return PinnedSnapshot(target, data)


def acquire_pinned(pin: Pin, directory: Path, *, fetcher: Callable[[str], bytes] = fetch_url) -> Path:
    """Use a local pin or download it, verifying its exact bytes before use."""
    return _acquire_pinned_snapshot(pin, directory, fetcher=fetcher).path


def input_path(pin: Pin, expected: Path, download_directory: Path | None) -> PinnedSnapshot:
    if expected.is_file():
        return PinnedSnapshot(expected, _read_pinned(pin, expected))
    if download_directory is None:
        raise FileNotFoundError(f"missing pinned input {expected}; provide --download-dir to acquire it")
    return _acquire_pinned_snapshot(pin, download_directory)


def glossary(source: Path | bytes) -> dict[str, dict[str, object]]:
    from carmel.benchmarks.structure import canonical

    output = subprocess.check_output(["pdftotext", "-layout", "-", "-"], input=_source_bytes(source))
    text = output.decode() if isinstance(output, bytes) else output
    rows: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3].startswith("InChI="):
            formula = {element: int(count or 1) for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", parts[1])}
            try:
                smiles = canonical(parts[2])
            except ValueError:
                smiles = None
            rows[parts[0].replace("\u2010", "-")] = {
                "smiles": smiles,
                "formula": formula,
                "source": "author-glossary",
            }
    return rows


def thermo(source: Path | bytes) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for line in _source_bytes(source).decode().splitlines():
        if len(line) > 79 and line[79] == "1":
            formula = {}
            for start in range(24, 44, 5):
                element = line[start : start + 2].strip().capitalize()
                if not element:
                    continue
                value = int(line[start + 2 : start + 5].strip() or "0")
                if value:
                    formula[element] = value
            rows[line[:18].split()[0]] = formula
    return rows


def inventory(
    labels: list[str], dictionary: dict[str, dict[str, object]], formulas: dict[str, dict[str, int]] | None = None
) -> list[dict[str, object]]:
    from carmel.benchmarks.structure import features, ketohydroperoxide

    records = []
    for label in labels:
        source = dictionary.get(label)
        if source and not source["smiles"]:
            source = None
        if source is None:
            smiles = ketohydroperoxide(label)
            if smiles and label.startswith("N") and label[1:] in dictionary:
                require(
                    smiles == dictionary[label[1:]]["smiles"],
                    f"grammar conflicts with author glossary: {label}",
                )
            source = {"smiles": smiles, "source": "tested-label-grammar: N?CnKETij"} if smiles else None
        empirical = formulas[label] if formulas else dict(features(source["smiles"]).atoms)  # type: ignore[index]
        if source and dict(features(source["smiles"]).atoms) != empirical:  # type: ignore[index]
            source = None
        records.append(
            dict(
                label=label,
                status="RESOLVED" if source else "UNRESOLVED",
                smiles=source["smiles"] if source else None,
                formula=empirical,
                source=source["source"] if source else "LLNL-thermo: formula only; no confident structure",
            )
        )
    return records


def candidate_reactions(reactions: list[dict[str, object]], species: dict[str, dict[str, object]]):
    for reaction in reactions:
        for side in (reaction["reactants"], reaction["products"]):  # type: ignore[union-attr]
            if len(side) == 1 and species[side[0]]["formula"] in (  # type: ignore[index]
                {"C": 7, "H": 15, "O": 4},
                {"C": 7, "H": 14, "O": 3},
            ):
                yield reaction
                break


def check_and_map(old_reactions, new_reactions, old_species, new_dictionary):
    from carmel.benchmarks.structure import classify, features, reaction_signature

    old = {s["label"]: s for s in old_species}
    for species in old_species:
        if species["formula"] in ({"C": 7, "H": 15, "O": 4}, {"C": 7, "H": 14, "O": 3}):
            require(
                species["status"] == "RESOLVED",
                f"NEEDS-INPUT common structural footing: {species}",
            )
    old_signatures = set()
    for reaction in old_reactions:
        if all(old[s]["smiles"] for s in (*reaction["reactants"], *reaction["products"])):  # type: ignore[index]
            left = tuple(old[s]["smiles"] for s in reaction["reactants"])  # type: ignore[index]
            right = tuple(old[s]["smiles"] for s in reaction["products"])  # type: ignore[index]
            old_signatures.add(reaction_signature(left, right))
    members = {f"G{i}": [] for i in range(1, 7)}
    new = {
        s: dict(formula=dict(features(r["smiles"]).atoms) if r["smiles"] else r["formula"], smiles=r["smiles"])
        for s, r in new_dictionary.items()
    }
    for reaction in candidate_reactions(new_reactions, new):
        left = tuple(new[s]["smiles"] for s in reaction["reactants"])
        right = tuple(new[s]["smiles"] for s in reaction["products"])
        classes = classify(left, right)
        if not classes:
            continue
        signature = reaction_signature(left, right)

        def formula_key(names, records):
            return sorted(tuple(sorted(records[s]["formula"].items())) for s in names)

        for prior in old_reactions:
            for prior_source, prior_target in (
                (prior["reactants"], prior["products"]),
                (prior["products"], prior["reactants"]),
            ):
                for source, target in (
                    (reaction["reactants"], reaction["products"]),
                    (reaction["products"], reaction["reactants"]),
                ):
                    if len(source) != len(prior_source) or len(target) != len(prior_target):
                        continue
                    if formula_key(source, new) != formula_key(prior_source, old) or formula_key(
                        target, new
                    ) != formula_key(prior_target, old):
                        continue
                    require(
                        all(old[s]["smiles"] for s in (*prior_source, *prior_target)),
                        "baseline member contains an unresolved species",
                    )
                    require(
                        reaction_signature(
                            tuple(old[s]["smiles"] for s in prior_source), tuple(old[s]["smiles"] for s in prior_target)
                        )
                        != signature,
                        "baseline reaction signature is already present",
                    )
        require(signature not in old_signatures, "candidate reaction signature is already in the baseline")
        for cls in classes:
            members[cls].append(dict(**reaction, absence="structural-signature-and-exhaustive-formula-screen"))
    return members, len(old_signatures)


def validate_ground_truth_formula(ground_truth: GroundTruth) -> None:
    require(
        dict(ground_truth.reactant_formula) == expected_reactant_formula(ground_truth.id),
        f"reactant formula is not frozen for {ground_truth.id}",
    )


def frozen_split(source_report: Path, census_root: Path) -> dict[str, object]:
    import yaml

    source = json.loads(_read_pinned_root_file(source_report))
    ledger = {
        r["path"]: r for r in json.loads(_read_pinned_root_file(census_root / "evidence/chemked-local-ledger.json"))
    }
    studies = []
    for study in source["studies"]:
        files = []
        for path in study["paths"]:
            relative = path.removeprefix("chemked/")
            raw = _read_pinned_root_file(census_root / path)
            require(
                hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == ledger[relative]["git_blob"],
                f"ChemKED Git blob changed: {path}",
            )
            data = yaml.safe_load(raw)
            points = data["datapoints"]
            temperatures = [float(p["temperature"][0].split()[0]) for p in points]
            ideal = list(range(len(points))) if data["apparatus"]["kind"] == "rapid compression machine" else []
            files.append(
                dict(
                    path=relative,
                    git_blob=ledger[relative]["git_blob"],
                    sha256=hashlib.sha256(raw).hexdigest(),
                    points=len(points),
                    below_750=sum(t < 750 for t in temperatures),
                    from_750_to_900=sum(750 <= t <= 900 for t in temperatures),
                    rcm_ideal_points=ideal,
                )
            )
        require(sum(f["points"] for f in files) == study["points"], f"split point count changed: {study['study']}")
        studies.append(dict(id=study["study"], doi=study["doi"], set=study["assignment"], files=files))
    return {
        "repository": "pr-omethe-us/ChemKED-database",
        "commit": source["source_commit"],
        "frozen_at": source["created_utc"],
        "studies": studies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census-root", type=Path, required=True, help="checkout of the pinned census inputs")
    parser.add_argument("--support-root", type=Path, required=True, help="directory containing pinned support files")
    parser.add_argument("--split-report", type=Path, required=True, help="pinned whole-study split report")
    parser.add_argument("--download-dir", type=Path, help="download missing mechanism/support pins here")
    parser.add_argument(
        "--contract", type=Path, default=REPO / "benchmarks/n-heptane-low-t/contract.json", help="contract to audit"
    )
    args = parser.parse_args()
    contract = load_contract(args.contract)
    old_snapshot = input_path(
        contract.mechanisms["v_old"], args.census_root / "evidence/nc7-v31.txt", args.download_dir
    )
    new_snapshot = input_path(
        contract.mechanisms["v_new"], args.census_root / "evidence/nc7-2016.txt", args.download_dir
    )
    glossary_snapshot = input_path(
        contract.glossary_source, args.support_root / contract.glossary_source.filename, args.download_dir
    )
    thermo_snapshot = input_path(
        contract.thermo_source, args.support_root / contract.thermo_source.filename, args.download_dir
    )
    old_labels, old_reactions = mechanism(old_snapshot.data)
    new_labels, new_reactions = mechanism(new_snapshot.data)
    dictionary = glossary(glossary_snapshot.data)
    require(set(new_labels) == dictionary.keys(), "new mechanism labels differ from glossary labels")
    old_inventory = inventory(old_labels, dictionary, thermo(thermo_snapshot.data))
    members, old_signatures = check_and_map(old_reactions, new_reactions, old_inventory, dictionary)
    for key, labels, reactions in (
        ("v_old", old_labels, old_reactions),
        ("v_new", new_labels, new_reactions),
    ):
        pin = contract.mechanisms[key]
        require(
            pin.unique_species == len(labels) and pin.reaction_entries == len(reactions),
            f"mechanism counts changed for {key}",
        )
    for source, snapshot in zip(
        (contract.glossary_source, contract.thermo_source), (glossary_snapshot, thermo_snapshot), strict=True
    ):
        source.verify(snapshot.data)
    require(
        [s.model_dump() for s in contract.old_species] == old_inventory,
        "old species inventory differs from the frozen contract",
    )
    for ground_truth in contract.ground_truth:
        validate_ground_truth_formula(ground_truth)
        require(
            [m.model_dump(mode="json") for m in ground_truth.members]
            == json.loads(json.dumps(members[ground_truth.id])),
            f"ground-truth member list changed for {ground_truth.id}",
        )
        print(
            f"PASS {ground_truth.id}: {ground_truth.member_count} member entries; "
            f"{ground_truth.resolved_count} resolved entries; "
            f"{ground_truth.resolved_species_count}/{ground_truth.member_species_count} resolved unique species; "
            "structural absence verified"
        )
    expected = inventory([s.label for s in contract.new_species], dictionary)
    require(
        [s.model_dump() for s in contract.new_species] == expected,
        "new species inventory differs from the frozen contract",
    )
    require(
        contract.split.model_dump(mode="json") == frozen_split(args.split_report, args.census_root),
        "frozen split differs from the contract",
    )
    totals = load_split(contract, args.census_root / "chemked")
    print("PASS split:", json.dumps(totals, sort_keys=True))
    print(
        f"PASS mechanism pins: {len(old_labels)}/{len(old_reactions)} old; {len(new_labels)}/{len(new_reactions)} new"
    )
    print(
        f"PASS old audit: {len(old_labels)} thermochemistry formulas; "
        f"{sum(s['status'] == 'RESOLVED' for s in old_inventory)} established graphs; "
        f"{old_signatures} fully mapped signatures; all potentially matching partial equations excluded by formulas"
    )
    print(
        "IDENTITY: UNVERIFIED; exact v3.1 header is March2012; Zhang2016 builds on updated "
        "NUIG base/pentane/n-hexane work, not an established v3.1 parent"
    )


if __name__ == "__main__":
    main()
