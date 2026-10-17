"""Small CHEMKIN inventory parser shared by frozen benchmark reproducers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from carmel.benchmarks.contract import read_pinned_file


class ChemkinReaction(TypedDict):
    equation: str
    line: int
    reactants: tuple[str, ...]
    products: tuple[str, ...]


def mechanism(source: Path | bytes) -> tuple[list[str], list[ChemkinReaction]]:
    """Return unique species labels and kinetic reaction entries from mechanism bytes."""
    display = str(source) if isinstance(source, Path) else "<verified snapshot>"
    raw_source = source if isinstance(source, bytes) else read_pinned_file(source)
    mode = None
    species: list[str] = []
    reactions: list[ChemkinReaction] = []
    for number, raw in enumerate(raw_source.decode(errors="replace").splitlines(), 1):
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
                        raise ValueError(f"unparsed reaction coefficient: {display}:{number}")
                    labels.extend([coefficient[2]] * int(coefficient[1] or 1))
                sides.append(tuple(labels))
            reactions.append(ChemkinReaction(equation=equation, line=number, reactants=sides[0], products=sides[1]))
    return sorted(set(species)), reactions
