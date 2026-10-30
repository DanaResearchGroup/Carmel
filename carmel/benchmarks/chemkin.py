"""Small CHEMKIN inventory parser shared by frozen benchmark reproducers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TypedDict

from carmel.benchmarks.contract import FrozenModel, read_pinned_file


class NormalizationRewrite(FrozenModel):
    """One faithful CHEMKIN numeric-token or card-format normalization."""

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
_NASA_NUMBER = re.compile(rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def _nasa7_fourth_card(previous: list[bytes], code: bytes) -> bytes:
    """Restore an absent index, or move a lone column-81 index to column 80."""
    absent = 60 <= len(code) <= 80 and not code[60:].strip(b" ")
    displaced = len(code) == 81 and not code[60:80].strip(b" ") and code[80:] == b"4"
    if len(previous) < 3 or not (absent or displaced):
        return code
    header, second, third = [line.partition(b"!")[0].rstrip(b"\r\n") for line in previous[-3:]]
    if any(card[79:80] != index for card, index in zip((header, second, third), (b"1", b"2", b"3"), strict=True)):
        return code
    if not header[:18].strip() or header[44:45] not in (b"G", b"L", b"S"):
        return code
    temperatures = [header[45:55], header[55:65]]
    if header[65:75].strip(b" "):
        temperatures.append(header[65:75])
    fields = temperatures + [
        card[offset : offset + 15]
        for card, count in ((second, 5), (third, 5), (code, 4))
        for offset in range(0, count * 15, 15)
    ]
    if any(_NASA_NUMBER.fullmatch(field.strip(b" ")) is None for field in fields):
        return code
    if second[75:79].strip(b" ") or third[75:79].strip(b" "):
        return code
    return code[:79].ljust(79, b" ") + b"4"


def _numeric_spans(code: bytes, section: bytes, previous: list[bytes]) -> list[tuple[int, int, bool]]:
    """Locate numeric fields without interpreting any part of a species label."""
    if section == b"THERMO":
        if code[79:80] == b"1" and code[44:45] in (b"G", b"L", b"S"):
            return [(start, start + 10, True) for start in (45, 55, 65)]
        # Coefficient cards must immediately follow a header/card sequence.
        for count in (1, 2, 3):
            if len(previous) < count:
                continue
            cards = previous[-count:]
            if cards[0][44:45] not in (b"G", b"L", b"S"):
                continue
            if all(card[79:80] == str(index).encode() for index, card in enumerate(cards, 1)):
                return [(start, start + 15, True) for start in range(0, 75, 15)]
        return []
    if section != b"REACTIONS":
        return []
    auxiliary = re.fullmatch(rb"\s*(LOW|HIGH|TROE|SRI|PLOG|REV|CHEB|TCHEB|PCHEB|LT|RLT)\s*/([^/]*)/\s*", code, re.I)
    if auxiliary:
        offset = auxiliary.start(2)
        return [(offset + token.start(), offset + token.end(), False) for token in re.finditer(rb"\S+", auxiliary[2])]
    tokens = list(re.finditer(rb"\S+", code))
    if len(tokens) < 4:
        return []
    parameters = tokens[-3:]
    equation = code[: parameters[0].start()].rstrip()
    if b"=" not in equation or equation.endswith((b"=", b">", b"+")):
        return []
    if not all(_NASA_NUMBER.fullmatch(token[0]) or _FORTRAN_ELESS_EXPONENT.fullmatch(token[0]) for token in parameters):
        return []
    return [(token.start(), token.end(), False) for token in parameters]


def normalize_for_ck2yaml(source: bytes, *, file: str) -> NormalizedInput:
    """Normalize numeric fields and absent/displaced NASA-7 fourth-card indices.

    CHEMKIN's Fortran reader accepts a whole numeric token such as
    ``1.5270+104`` as ``1.5270E+104``. Cantera's ck2yaml does not.
    Numeric fields are scoped to NASA temperatures/coefficients and reaction
    rate parameters. Fixed-width fields retain their width; fields without
    room for the exponent letter remain refused by conversion. Identifiers,
    header text and comments are never rewritten. A fourth card with a blank
    trailer or lone column-81 ``4`` is repaired after three valid NASA cards.
    """
    rewrites: list[NormalizationRewrite] = []
    normalized_lines: list[bytes] = []
    section = b""
    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        body = line.rstrip(b"\r\n")
        ending = line[len(body) :]
        content, bang, note = body.partition(b"!")
        words = content.split()
        head = words[0].upper() if words else b""
        if head in (b"THERMO", b"THERM"):
            section = b"THERMO"
        elif head in (b"REACTIONS", b"REAC"):
            section = b"REACTIONS"
        elif head in (b"SPECIES", b"SPEC", b"ELEMENTS", b"ELEM", b"END"):
            section = b""
        pieces: list[bytes] = []
        cursor = 0
        for start, stop, fixed in _numeric_spans(content, section, normalized_lines):
            field = content[start:stop]
            match = _FORTRAN_ELESS_EXPONENT.fullmatch(field.strip(b" "))
            if match is None:
                continue
            before = field if fixed else match[0]
            after = match.group(1) + b"E" + match.group(2)
            if fixed:
                if len(after) > len(field):
                    continue
                after = after.rjust(len(field), b" ")
            rewrites.append(
                NormalizationRewrite(
                    file=file,
                    line=line_number,
                    before=before.decode("ascii"),
                    after=after.decode("ascii"),
                )
            )
            pieces.extend((content[cursor:start], after))
            cursor = stop
        content = b"".join([*pieces, content[cursor:]])
        if section == b"THERMO":
            repaired = _nasa7_fourth_card(normalized_lines, content)
            if repaired != content:
                rewrites.append(
                    NormalizationRewrite(
                        file=file,
                        line=line_number,
                        before=content.decode("ascii"),
                        after=repaired.decode("ascii"),
                    )
                )
                content = repaired
        normalized_lines.append(content + bang + note + ending)
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
