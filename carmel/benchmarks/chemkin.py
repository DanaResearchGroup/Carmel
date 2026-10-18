"""Small CHEMKIN inventory parser shared by frozen benchmark reproducers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TypedDict

from carmel.benchmarks.contract import FrozenModel, read_pinned_file


class NormalizationRewrite(FrozenModel):
    """One faithful CHEMKIN numeric-token normalization."""

    file: str
    line: int
    before: str
    after: str


class NormalizedInput(FrozenModel):
    """Pinned raw and normalized bytes supplied to ck2yaml."""

    data: bytes
    raw_sha256: str
    normalized_sha256: str
    rewrites: tuple[NormalizationRewrite, ...]


_FORTRAN_ELESS_EXPONENT = re.compile(rb"(?<![A-Za-z0-9_.])([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)(?![A-Za-z0-9_.])")


def normalize_for_ck2yaml(source: bytes, *, file: str) -> NormalizedInput:
    """Make CHEMKIN's Fortran E-less exponents explicit outside comments.

    CHEMKIN's Fortran reader accepts a whole numeric token such as
    ``1.5270+104`` as ``1.5270E+104``. Cantera's ck2yaml does not. This
    byte-preserving normalization inserts only the omitted exponent letter;
    text after a line's first ``!`` and tokens embedded in labels are retained
    exactly.
    """
    rewrites: list[NormalizationRewrite] = []
    normalized_lines: list[bytes] = []
    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        code, marker, comment = line.partition(b"!")

        def replace(match: re.Match[bytes], *, current_line: int = line_number) -> bytes:
            before = match.group(0)
            after = match.group(1) + b"E" + match.group(2)
            rewrites.append(
                NormalizationRewrite(
                    file=file,
                    line=current_line,
                    before=before.decode("ascii"),
                    after=after.decode("ascii"),
                )
            )
            return after

        normalized_lines.append(_FORTRAN_ELESS_EXPONENT.sub(replace, code) + marker + comment)
    normalized = b"".join(normalized_lines)
    return NormalizedInput(
        data=normalized,
        raw_sha256=hashlib.sha256(source).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized).hexdigest(),
        rewrites=tuple(rewrites),
    )


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
