"""Triage tool: find glyphs whose embedded OUTLINE contradicts the character the
extraction pipeline decodes them to -- the symbol-font-Latin impostor fault.

This exists because the fault is invisible to every cheap signal. A symbol font can hold
an en-dash in its ``e`` slot and a phi in its ``f`` slot; the byte decodes to a valid
Latin ``e``/``f`` by the base encoding, the font carries no ``/ToUnicode`` and no
``/Differences`` to contradict it, its ``/Flags`` say Nonsymbolic, and even the embedded
program's own charset NAMES the glyph ``e``/``f``. Every name and flag AGREES with the
wrong character. The ONLY thing that disagrees is the drawn ink, so the only way to find
these is to read the outline -- which the extraction path (pypdf-only, by design) does
not do, and which building a general glyph classifier is explicitly not this project's job.

So this module is a TRIAGE tool, not an oracle, and the distinction is load-bearing:

* It ACCUSES, it does not CONVICT. Reference-free geometric flags false-positive on
  ordinary type -- an italic ``f`` and a ``J`` and a ``Q`` all descend, a small-caps
  ``e`` can be a single contour. The flags below are tuned to catch the known impostors
  without drowning, but a flagged glyph is a CANDIDATE for a human to confirm against the
  actual page, never a fact.
* It NEVER writes a repair. It emits :class:`ImpostorCandidate` records carrying the
  evidence (bbox, contour count, decoded character, program sha256); a human reads that
  evidence, confirms the glyph against the rendered page, and -- only then -- hand-authors
  a :class:`carmel.services.pdf_fragments.GlyphRepair` entry, scoped by document + program
  + name exactly as every other entry is. There is deliberately no "suggested replacement"
  field: identifying the fault is an outline measurement; naming the character is a human
  judgement this tool must not pre-empt.
* It is DEV/OFFLINE only. ``fontTools`` (the outline reader) is a dev dependency, imported
  lazily here, and no runtime ``carmel`` module imports this one. The extraction path
  never gains an outline parser; this runs beside it, over the registry, to size the fault.

Run it with :func:`census`, or ``python -m carmel.services.glyph_impostor_census <pdf>...``.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The em square CFF charstrings are expressed in. Thresholds below are fractions of it, so
# they hold regardless of a font's declared units-per-em (CFF is 1000 by construction, but
# reading it rather than assuming it keeps the tool honest if a program says otherwise).
_DEFAULT_UPM = 1000

# Lowercase letters that never descend below the baseline in normal type. A deep descender
# on one of these, with a closed counter, is the phi-as-'f' signature and its kin.
_LOWERCASE_NO_DESCENDER = set("abcdefhiklmnorstuvwxz")


@dataclass(frozen=True)
class ImpostorCandidate:
    """One glyph whose outline the tool flags as possibly not the character it decodes to.

    Every field is EVIDENCE, read from the document, for a human to confirm or dismiss.
    There is no replacement and no verdict: this record says "look at this glyph", not
    "this glyph is X".
    """

    document_sha256: str
    font_program_sha256: str
    font_base_name: str
    decoded_char: str
    """The character the base encoding decodes this glyph's code to -- what the pipeline
    currently believes, and what the outline may contradict."""

    glyph_name: str
    """The embedded program's own name for the glyph. Often AGREES with ``decoded_char``
    (a symbol subset names its en-dash glyph ``e``); recorded because that agreement is
    itself part of why the fault is invisible."""

    bbox: tuple[int, int, int, int]
    contours: int
    advance_width: int | None
    units_per_em: int
    flags: tuple[str, ...]
    """Which conservative signals tripped (``HORIZONTAL_BAR``, ``COUNTERED_DESCENDER``).
    A non-empty tuple means "candidate"; the flags are triage hints, not conclusions."""

    drawn_count: int | None = None
    """How many times this glyph is actually SHOWN in the document (best effort; ``None``
    if the content-stream pass could not run). A charset candidate that is never drawn
    corrupts no extracted value; one drawn many times sizes the damage. This is the number
    a sizing pass needs -- a glyph in a font subset is a latent fault, a drawn glyph is a
    live one."""

    def summary(self) -> str:
        x0, y0, x1, y1 = self.bbox
        adv = "?" if self.advance_width is None else str(self.advance_width)
        drawn = "?" if self.drawn_count is None else str(self.drawn_count)
        return (
            f"{self.font_base_name} prog={self.font_program_sha256[:12]} "
            f"glyph {self.glyph_name!r} decodes to {self.decoded_char!r} drawn x{drawn} | "
            f"bbox=[{x0},{x1}]x[{y0},{y1}] contours={self.contours} adv={adv} "
            f"em={self.units_per_em} flags={','.join(self.flags)}"
        )


@dataclass
class DocumentCensus:
    document_sha256: str
    programs_scanned: int
    glyphs_scanned: int
    candidates: list[ImpostorCandidate] = field(default_factory=list)


def _outline_flags(decoded_char: str, bbox: tuple[int, int, int, int], contours: int, upm: int) -> tuple[str, ...]:
    """The conservative triage flags for one glyph outline. Tuned to catch the known
    impostor SHAPES while leaving ordinary type mostly alone -- but see the module
    docstring: this accuses, it does not convict, and false positives are expected and
    acceptable because a human confirms every candidate.

    * ``HORIZONTAL_BAR`` -- a single-contour glyph that is a wide, short bar floating off
      the baseline: an en-dash / minus / rule mis-decoded to a letter. No Latin letter is
      a thin bar at mid-height, so this signal is strong.
    * ``COUNTERED_DESCENDER`` -- a lowercase non-descending letter whose outline has a
      closed counter (>=2 contours) AND descends well below the baseline: the phi-as-'f'
      signature. The counter requirement is what separates it from a genuine italic ``f``,
      which descends but is a single open stroke.
    """
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    flags: list[str] = []
    if contours == 1 and height < 0.25 * upm and y0 > 0.12 * upm and width > 0.30 * upm:
        flags.append("HORIZONTAL_BAR")
    if decoded_char in _LOWERCASE_NO_DESCENDER and contours >= 2 and y0 < -0.06 * upm:
        flags.append("COUNTERED_DESCENDER")
    return tuple(flags)


def _iter_cff_programs(pdf_bytes: bytes) -> list[tuple[str, str, bytes]]:
    """``(resource_font_program_sha256, base_font_name, cff_bytes)`` for every embedded
    CFF (``/FontFile3``) program in the document, de-duplicated by sha256. TrueType
    (``/FontFile2``) and Type1 (``/FontFile``) are out of scope for this tool: the impostor
    fault is measured only where the outline can be read, and CFF is where it was found."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    seen: dict[str, tuple[str, bytes]] = {}
    for page in reader.pages:
        try:
            _ = page.mediabox
        except Exception:  # noqa: BLE001 - a page we cannot read is a page we skip
            continue
        resources = page.get("/Resources")
        if resources is None:
            continue
        fonts = resources.get_object().get("/Font")
        if fonts is None:
            continue
        for _key, ref in fonts.get_object().items():
            font_obj = ref.get_object()
            descriptor = font_obj.get("/FontDescriptor")
            if descriptor is None:
                continue
            program_ref = descriptor.get_object().get("/FontFile3")
            if program_ref is None:
                continue
            data = bytes(program_ref.get_object().get_data())
            sha = hashlib.sha256(data).hexdigest()
            if sha not in seen:
                seen[sha] = (str(font_obj.get("/BaseFont", "?")), data)
    return [(sha, base, data) for sha, (base, data) in seen.items()]


def _operand_bytes(operand: Any) -> bytes:
    """The raw show-operand bytes, whichever string type pypdf parsed it into. A ``(e)Tj``
    operand becomes a ``TextStringObject`` (a ``str`` subclass) when it decodes as text and
    a ``ByteStringObject`` (``bytes``) otherwise; both carry ``.original_bytes``, which is
    the code the ``Tj`` actually drew and the identity drawn counts must key on."""
    original = getattr(operand, "original_bytes", None)
    if isinstance(original, (bytes, bytearray)):
        return bytes(original)
    if isinstance(operand, (bytes, bytearray)):
        return bytes(operand)
    if isinstance(operand, str):
        return operand.encode("latin-1", "ignore")
    return b""


def _drawn_counts(pdf_bytes: bytes) -> dict[tuple[str, int], int]:
    """Best-effort ``(font_program_sha256, byte_code) -> times shown`` across the document.

    Walks each page's content stream tracking the current font (``Tf``) and counting the
    bytes of every ``Tj``/``TJ`` operand. Byte code, not decoded character: it is the code
    the ``Tj`` operand actually carries, so it lines up with what draws the glyph. Returns
    an empty map on any failure -- drawn counts are an enrichment, never a correctness
    dependency, so a document whose streams will not parse still yields its charset
    candidates with ``drawn_count`` left ``None``."""
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:  # noqa: BLE001
        return {}
    counts: dict[tuple[str, int], int] = {}
    for page in reader.pages:
        try:
            resources = page.get("/Resources")
            fonts = resources.get_object().get("/Font").get_object() if resources is not None else None
            if fonts is None:
                continue
            # resource font name -> embedded program sha256 (only fonts with a CFF program).
            name_to_sha: dict[str, str] = {}
            for key, ref in fonts.items():
                descriptor = ref.get_object().get("/FontDescriptor")
                if descriptor is None:
                    continue
                program = descriptor.get_object().get("/FontFile3")
                if program is None:
                    continue
                name_to_sha[str(key)] = hashlib.sha256(bytes(program.get_object().get_data())).hexdigest()
            content = page.get_contents()
            if content is None:
                continue
            from pypdf.generic import ContentStream

            operations = ContentStream(content, reader).operations
            current_sha: str | None = None
            for operands, operator in operations:
                if operator == b"Tf" and operands:
                    current_sha = name_to_sha.get(str(operands[0]))
                elif operator in (b"Tj", b"'", b'"') and current_sha is not None and operands:
                    for byte in _operand_bytes(operands[-1]):
                        counts[(current_sha, byte)] = counts.get((current_sha, byte), 0) + 1
                elif operator == b"TJ" and current_sha is not None and operands:
                    for element in operands[0] if isinstance(operands[0], ArrayObject) else []:
                        for byte in _operand_bytes(element):
                            counts[(current_sha, byte)] = counts.get((current_sha, byte), 0) + 1
        except Exception:  # noqa: BLE001 - one unparseable page does not sink the pass
            continue
    return counts


def scan_document(pdf_bytes: bytes) -> DocumentCensus:
    """Read every embedded CFF program's glyph outlines and return the candidates whose
    shape contradicts the Latin character their name/encoding claims. Pure measurement:
    reads the document, writes nothing, decides nothing beyond "worth a human's look"."""
    from fontTools.cffLib import CFFFontSet
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.pens.recordingPen import RecordingPen

    document_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    drawn = _drawn_counts(pdf_bytes)
    census = DocumentCensus(document_sha256=document_sha256, programs_scanned=0, glyphs_scanned=0)
    for program_sha, base_name, cff_bytes in _iter_cff_programs(pdf_bytes):
        try:
            cff = CFFFontSet()
            cff.decompile(io.BytesIO(cff_bytes), None)
            font = cff[cff.fontNames[0]]
            charstrings = font.CharStrings
            charset = font.charset
        except Exception:  # noqa: BLE001 - a program we cannot parse contributes no candidates
            continue
        census.programs_scanned += 1
        upm = _DEFAULT_UPM
        for glyph_name in charset:
            if glyph_name == ".notdef":
                continue
            # We can only judge a glyph the pipeline would decode to a Latin CHARACTER; the
            # program's charset name is that decode's proxy for a base-encoded subset (a
            # symbol font names its en-dash glyph 'e' precisely because byte 0x65 -> 'e').
            decoded_char = glyph_name if len(glyph_name) == 1 and glyph_name.isalpha() else None
            if decoded_char is None:
                continue
            census.glyphs_scanned += 1
            try:
                bounds_pen = BoundsPen(charstrings)
                charstrings[glyph_name].draw(bounds_pen)
                if bounds_pen.bounds is None:
                    continue
                bbox = tuple(int(round(v)) for v in bounds_pen.bounds)
                recording = RecordingPen()
                charstrings[glyph_name].draw(recording)
                contours = sum(1 for op, _ in recording.value if op == "moveTo")
                advance = getattr(charstrings[glyph_name], "width", None)
            except Exception:  # noqa: BLE001 - an unreadable outline is not a candidate
                continue
            flags = _outline_flags(decoded_char, bbox, contours, upm)  # type: ignore[arg-type]
            if flags:
                # The byte that draws this glyph in a base-encoded (WinAnsi) subset is the
                # code of its decoded character -- the same identity `drawn` is keyed on.
                drawn_count = drawn.get((program_sha, ord(decoded_char))) if drawn else None
                census.candidates.append(
                    ImpostorCandidate(
                        document_sha256=document_sha256,
                        font_program_sha256=program_sha,
                        font_base_name=base_name,
                        decoded_char=decoded_char,
                        glyph_name=glyph_name,
                        bbox=bbox,  # type: ignore[arg-type]
                        contours=contours,
                        advance_width=int(advance) if advance is not None else None,
                        units_per_em=upm,
                        flags=flags,
                        drawn_count=drawn_count,
                    )
                )
    return census


def census(paths: list[Path]) -> list[DocumentCensus]:
    """Scan several documents and return one :class:`DocumentCensus` each. Missing files
    are skipped with a census carrying zero programs, so a caller can report them."""
    results: list[DocumentCensus] = []
    for path in paths:
        if not path.exists():
            results.append(DocumentCensus(document_sha256="(absent)", programs_scanned=0, glyphs_scanned=0))
            continue
        results.append(scan_document(path.read_bytes()))
    return results


def format_report(results: list[tuple[str, DocumentCensus]]) -> str:
    """A human-readable census: per document, the flagged candidates with their evidence,
    and a total. Reminds the reader at the top that these are candidates, not conclusions."""
    lines = [
        "Symbol-font-Latin impostor census -- CANDIDATES for human confirmation, not verdicts.",
        "Each flagged glyph's OUTLINE contradicts the character its name/encoding claims;",
        "confirm against the rendered page before authoring any repair entry.",
        "",
    ]
    total = 0
    for name, doc in results:
        lines.append(f"### {name}  ({doc.programs_scanned} CFF programs, {doc.glyphs_scanned} latin-named glyphs)")
        if not doc.candidates:
            lines.append("    (no candidates)")
        for candidate in sorted(doc.candidates, key=lambda c: (c.decoded_char, c.font_base_name)):
            lines.append(f"    {candidate.summary()}")
            total += 1
        lines.append("")
    lines.append(f"TOTAL candidates: {total}")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print("usage: python -m carmel.services.glyph_impostor_census <pdf> [<pdf> ...]")
        return 2
    results = [(p.name, doc) for p, doc in zip(paths, census(paths), strict=True)]
    print(format_report(results))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    raise SystemExit(_main(sys.argv[1:]))
