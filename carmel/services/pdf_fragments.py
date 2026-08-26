"""Text-show fragments with absolute page geometry, recovered from a PDF.

This is the substrate beneath M1's ``TABLE_CELL`` locators. Validator V7 refuses a
``CharSpanLocator`` as the source of a series VALUE, so a ``DatasetEnvelope`` stays
unconstructible until something can address a value BY CELL -- and a cell is built
out of the fragments this module returns.

It stops deliberately short of grouping. Nothing here decides that two fragments
share a word, a row, a column or a cell. Grouping is a DERIVED structural claim and
the adversarial core of M1; extraction is mechanical. Keeping them in separate
modules is what stops a fabricated pairing from riding in on the back of a
mechanical step, the same way span stitching was separated in the condition-set lane.

Why the engine below looks the way it does -- every one of these was established by
running the alternatives against real publisher PDFs, and each obvious route fails:

* ``extract_text(extraction_mode="layout")`` pads with runs of spaces so that column
  identity is implied by whitespace WIDTH. That is structure inferred from prose,
  which the P0-c ruling outlawed. It also truncates.
* ``visitor_text`` is the obvious API and it is a TRAP: it fires once per LINE and
  reports only that line's STARTING x. Three columns arrive merged into one fragment
  with the per-column x already gone, so a caller is left re-splitting on whitespace
  -- reintroducing the same outlawed fabrication one layer down while believing it
  holds real geometry.
* ``text_show_operations`` returns ``BTGroup``s that are ALSO one-per-line, and it
  left-aligns the whole page by subtracting ``min(tx)`` before returning. Its return
  value therefore carries no absolute coordinates at all.

What does work is the per-text-show ``TextStateParams`` list that pypdf's layout-mode
engine builds internally and then discards. Reaching for it means depending on
private API (see :func:`_engine`), which is guarded rather than assumed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata
import io
import logging
import math
import re
import zlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "FragmentAvailability",
    "FragmentExtraction",
    "FragmentPageFailure",
    "GlyphMapping",
    "GlyphRepair",
    "TextFragment",
    "extract_fragments",
    "registry_glyph_repairs_for_document",
]

logger = logging.getLogger(__name__)


class GlyphMapping(StrEnum):
    """Whether a fragment's glyphs decoded to real characters.

    Named ``GlyphMapping`` rather than the more obvious ``GlyphHealth`` because
    :class:`carmel.services.numeric.GlyphHealth` already exists and means something
    DIFFERENT: a document-level corruption assessment used to quarantine numerals.
    Two same-named types at different scopes would eventually be passed to each
    other's call sites, and the mistake would typecheck under ``Any``.

    This is a flag, never a repair. See :data:`_UNMAPPED_MARKER_RE`.
    """

    MAPPED = "mapped"
    """Every glyph decoded to a character. Says nothing about whether that character
    is the RIGHT one -- a PDF with a broken ``ToUnicode`` map can decode ``+`` as
    ``þ`` and ``=`` as ``¼`` perfectly "successfully"."""

    UNMAPPED = "unmapped"
    """At least one glyph had no usable mapping and surfaced as a marker rather than
    a character. The fragment's text is returned UNMODIFIED; a consumer that treats
    it as a value is reading a marker as data."""

    REPAIRED = "repaired"
    """Every glyph decoded, but at least one of them only through Carmel's own
    evidence-scoped verdict registry (:data:`_GLYPH_VERDICTS`) rather than through the
    document. A third member instead of a promotion to :attr:`MAPPED`, deliberately:
    the document never said what this glyph means -- Carmel concluded it, from
    recorded evidence, under a gate scoped to one embedded font program and one glyph
    code (and, where the character needs document context, one document) -- and an
    artifact built from the fragment must be able to say so. Consumers that admit it
    are admitting that conclusion by name, never by accident: everything else in this
    codebase that switches on this enum treats any member it does not explicitly admit
    as refusable."""

    UNRESOLVED_IMPOSTOR = "unresolved_impostor"
    """A glyph the registry has LOOKED AT and declined to decide: a known symbol-font
    impostor whose true character the embedded outline does not pin. Stronger than
    silence -- an unregistered mis-decode surfaces :attr:`MAPPED` (nothing flagged it)
    and an unregistered marker surfaces :attr:`UNMAPPED`; this member is reserved for a
    glyph a :class:`GlyphRefusal` verdict names explicitly, on recorded evidence, as an
    impostor Carmel refuses to guess. The fragment's text is returned UNMODIFIED (the
    wrong Latin character the font handed back), so a consumer that reads it as data is
    reading a known-wrong character -- which is why this member is NOT admissible.

    It must never be added to ``carmel.services.pdf_tables._ADMISSIBLE_GLYPH_MAPPINGS``.
    That allowlist is the seam that makes an impostor inside a table box REFUSE with no
    change at the table layer; admitting this member there would ship the very
    mis-decoded value the verdict exists to withhold. It IS text, though (a printed
    character, not a marker), so the row-membership allowlists that ask "is this a
    printed line at all?" DO count it -- see
    ``carmel.services.pdf_tables._ROW_TEXT_GLYPH_MAPPINGS`` and
    ``carmel.services.pdf_cells``."""


# Markers a PDF text extractor emits when a glyph has no usable ``ToUnicode`` entry.
#
# This matters far more than it looks. In a real corpus rate-constant table the
# temperature exponent ``n = -1.0`` arrives with its MINUS SIGN as a separate
# fragment whose decoded text is ``/C0`` (an independent extractor renders the same
# glyph ``(cid:3)``). A consumer that ignores the marker reads ``+1.0``: a silent
# SIGN INVERSION inside an otherwise perfectly well-formed number. That is strictly
# worse than a missing value, because nothing downstream looks wrong.
#
# Flagged, and never repaired by DEFAULT. Mapping ``/C0`` to U+2212 -- or ``þ`` to
# ``+`` and ``¼`` to ``=``, which the same PDFs also need -- is a SEMANTIC claim about
# what the document meant, and grounding proves LOCATION, never MEANING. A repair
# needs its own gate and its own evidence. That registry now exists
# (:data:`_GLYPH_VERDICTS`): each entry pins ONE embedded font program and ONE glyph
# code (and, for a character that needs document context, ONE document's bytes),
# carries the recorded evidence it rests on, and is EITHER a :class:`GlyphRepair`
# (surfaces :attr:`GlyphMapping.REPAIRED` -- never ``MAPPED``, so nothing downstream can
# mistake Carmel's conclusion for the document's) OR a :class:`GlyphRefusal` (surfaces
# :attr:`GlyphMapping.UNRESOLVED_IMPOSTOR` -- a glyph looked at and declined, which
# refuses downstream instead of being read as its wrong Latin decode). A glyph with no
# entry at all stays flagged with its text unmodified, exactly as before.
# The `/C\d+` arm is DELIBERATELY bounded on both sides. An unanchored `/C\d+`
# substring match is catastrophic in a combustion codebase: it flags `/C2H4`, `C1/C2`,
# `H2/CO`-style species lists, appendix labels and file paths as corrupt. The lookarounds
# require the token to stand alone, so `/C0` matches while `/C2H4` (letter after) and
# `C1/C2` (digit before the slash) do not.
_UNMAPPED_MARKER_RE = re.compile(
    r"""
    \( cid: \d+ \)              # (cid:3)  -- raw character id, no mapping at all
    | (?<![0-9A-Za-z]) /C \d+ (?![0-9A-Za-z])   # /C0 -- a standalone glyph-NAME escape
    | �                    # U+FFFD   -- replacement character
    """,
    re.VERBOSE,
)


class GlyphVerdictScope(StrEnum):
    """How widely a :class:`GlyphVerdict` is asserted -- the keying decision made explicit
    rather than encoded in whether ``document_sha256`` happens to be set.

    The safety of a verdict comes from ``font_program_sha256`` + ``glyph_name``: together
    they pin the exact embedded program and the exact decoded piece, and hence the exact
    ink. ``document_sha256`` is a blast-radius limiter and an evidence-provenance guard,
    not the thing that makes the conclusion true. So a verdict whose character the OUTLINE
    pins independent of context can be asserted for every document embedding the program;
    one whose character needs document context must stay narrowed to the one document the
    evidence was read from.
    """

    FONT_PROGRAM = "font_program"
    """Context-free: matches any document embedding this program. ``document_sha256`` is
    absent -- there is no per-document claim to make -- and the document the evidence was
    read from is recorded in ``evidence`` for provenance only. Used where the outline alone
    pins the character (a phi's bowl-and-through-stroke) or declines it (a refusal)."""

    DOCUMENT = "document"
    """Narrowed: matches only the one document whose sha256 is ``document_sha256``. Used
    where the replacement chose using document context -- which dash a bar is (en-dash vs
    minus, by role), where a ring sits (a degree, by its superscript placement) -- so the
    conclusion is asserted for that file and no other."""


@dataclass(frozen=True, kw_only=True)
class GlyphVerdict:
    """One decided glyph: one embedded font program's one decoded piece, judged.

    **A global glyph-name mapping is silent semantic corruption and must never ship.** A
    ``/Differences`` name like ``/C14`` is a FONT-LOCAL code, not a character, and even a
    base-encoded ``f`` is a font-local claim once the font is a symbol subset: another
    embedded font is free to bind the same name or slot to anything at all, and a table
    keyed on the name alone would rewrite it everywhere while looking perfectly reasonable.
    So every entry is pinned by, and applies only when BOTH match:

    * ``font_program_sha256`` -- the sha256 of the embedded font program's decoded bytes
      (``FontFile``/``FontFile2``/``FontFile3``). The glyph's meaning lives in the program
      that draws it, so this is the identity the conclusion is actually ABOUT, and this is
      where the safety comes from: it pins the exact program, and hence the exact ink.
    * ``glyph_name`` -- the decoded per-glyph substring the fragment lane produces for the
      code, matched against ONE glyph piece at a time. For a glyph the document leaves
      unmapped this is the font's own ``/Differences`` name (``/C14``), an unmapped MARKER;
      for a glyph the document mis-decodes it is the wrong character the font hands back (a
      symbol font's ``f`` slot holds a phi, so the piece is a literal ``f``). The match is
      per PIECE, and a piece is exactly one glyph's decode -- ordinary text that merely
      SPELLS ``f`` or ``/C14`` arrives as several pieces and cannot match. That per-glyph
      partition, together with the program pin, is what makes matching on a bare ``f`` a
      font-program-and-code conclusion and NOT a global ``f`` -> phi substitution.

    ``scope`` records how widely the conclusion is asserted (see :class:`GlyphVerdictScope`).
    The old design made ``document_sha256`` a mandatory third gate and argued at length for
    it; that argument was right for the dash class, whose character DOES need document
    context, but wrong as a universal rule -- for a context-free character (a phi the outline
    pins) the document sha limits blast radius without adding truth, and keying on it forces
    one hand-authored entry per document, which does not scale past a handful. ``FONT_PROGRAM``
    is the principled widening; ``DOCUMENT`` is the narrowing the dash class still needs. Two
    entries that could both match the same (document, program, glyph) are a registry bug, not
    last-write-wins: :func:`_assert_no_overlapping_verdicts` rejects them at import.

    A concrete verdict is EITHER a :class:`GlyphRepair` (a replacement, surfaces
    :attr:`GlyphMapping.REPAIRED`) or a :class:`GlyphRefusal` (a reason, surfaces
    :attr:`GlyphMapping.UNRESOLVED_IMPOSTOR`). Where the evidence does not support a mapping,
    the honest entry is a refusal -- or, for a glyph not looked at at all, NO entry: the
    fragment then refuses downstream with the glyph named, a better outcome than a guess.
    """

    scope: GlyphVerdictScope
    font_program_sha256: str
    glyph_name: str
    evidence: str
    """What the conclusion rests on, recorded beside it so a reviewer can re-derive or
    refute it without an archaeology dig. State what was actually consulted."""

    font_base_name: str = ""
    """The font's ``/BaseFont`` at the time the evidence was read. Diagnostic only --
    subset prefixes are arbitrary and names are forgeable, so this never gates."""

    document_sha256: str | None = None
    """Present iff ``scope`` is :attr:`~GlyphVerdictScope.DOCUMENT`; the sha256 of the whole
    document's raw bytes it is narrowed to."""

    def __post_init__(self) -> None:
        if self.scope is GlyphVerdictScope.DOCUMENT and self.document_sha256 is None:
            raise ValueError(f"DOCUMENT-scoped verdict for {self.glyph_name!r} needs a document_sha256")
        if self.scope is GlyphVerdictScope.FONT_PROGRAM and self.document_sha256 is not None:
            raise ValueError(
                f"FONT_PROGRAM-scoped verdict for {self.glyph_name!r} must not carry a document_sha256 "
                "(the provenance document belongs in evidence, not the gate)"
            )

    def applies_to(self, document_sha256: str) -> bool:
        """Whether this verdict is in force for the document with this raw-bytes sha256."""
        return self.scope is GlyphVerdictScope.FONT_PROGRAM or self.document_sha256 == document_sha256


@dataclass(frozen=True, kw_only=True)
class GlyphRepair(GlyphVerdict):
    """A verdict that DECODES the glyph: its piece is replaced with ``replacement`` and the
    fragment surfaces :attr:`GlyphMapping.REPAIRED`.

    Two kinds of glyph reach a repair, surfacing differently on the way IN but identically on
    the way OUT. An UNMAPPED glyph (marker piece) is one the document never decoded; a
    MIS-DECODED glyph (valid-Latin piece) is one the document decoded to the wrong character
    -- a symbol font handing back ``f``/``e`` that no ``/ToUnicode`` and no ``/Differences``
    contradict, so nothing downstream flags it. The mis-decoded case is the more dangerous of
    the two precisely because it reads as clean data; it is caught only because the embedded
    program that draws it is pinned by sha256 and its glyph identified by outline (see each
    entry's evidence). A fragment whose markers are not ALL covered by matching repairs keeps
    its text unmodified and stays ``UNMAPPED`` -- a half-repaired string would read as data
    while still carrying a marker.
    """

    replacement: str


@dataclass(frozen=True, kw_only=True)
class GlyphRefusal(GlyphVerdict):
    """A verdict that DECLINES the glyph: a known impostor whose true character the outline
    does not pin. The piece is left UNMODIFIED and the fragment surfaces
    :attr:`GlyphMapping.UNRESOLVED_IMPOSTOR`, which is non-admissible, so the fragment refuses
    downstream instead of being read as its wrong Latin decode.

    A refusal is a DIFFERENT and stronger claim than silence. An unregistered mis-decode
    surfaces ``MAPPED`` and sails through; a refusal says a human looked at this program's
    glyph, on recorded evidence, and could not honestly name its character -- a bar that could
    be a tilde or an approx sign, a double rule that could be an equals or a parallel-to, a
    single descender that resembles a gamma without excluding its alternatives the way a phi's
    bowl-and-through-stroke excludes them. Better a refusal than the guess that produced the
    original phi-as-``f`` fault run in the other direction.
    """

    reason: str


def _assert_no_overlapping_verdicts(verdicts: tuple[GlyphVerdict, ...]) -> None:
    """Reject a registry where two entries could both match one (document, program, glyph).

    The match key is ``(font_program_sha256, glyph_name)``. Within a key, a FONT_PROGRAM entry
    matches EVERY document, so it overlaps any other entry for the same key; two DOCUMENT
    entries overlap iff they name the same document; anything else is disjoint. An overlap is a
    bug -- the application site narrows by document into a ``dict`` keyed on glyph name, which
    would otherwise resolve a collision by last-write-wins, silently picking one conclusion
    over another -- so it fails at import, loudly, naming the glyph.
    """
    by_key: dict[tuple[str, str], list[GlyphVerdict]] = {}
    for verdict in verdicts:
        by_key.setdefault((verdict.font_program_sha256, verdict.glyph_name), []).append(verdict)
    for (program, name), group in by_key.items():
        if len(group) == 1:
            continue
        if any(v.scope is GlyphVerdictScope.FONT_PROGRAM for v in group):
            raise ValueError(
                f"overlapping glyph verdicts for program {program[:12]} glyph {name!r}: a FONT_PROGRAM "
                "verdict matches every document, so it cannot coexist with another entry for the same glyph"
            )
        documents = [v.document_sha256 for v in group]
        if len(set(documents)) != len(documents):
            raise ValueError(
                f"overlapping glyph verdicts for program {program[:12]} glyph {name!r}: two DOCUMENT "
                "verdicts name the same document"
            )


#: Every glyph verdict this module will apply -- repairs and refusals both. Append-only;
#: the keying, scope and refuse-don't-guess policy are on :class:`GlyphVerdict`, and
#: :func:`_assert_no_overlapping_verdicts` (called just below) rejects any pair that could
#: both match one glyph. Dash and degree entries stay :attr:`GlyphVerdictScope.DOCUMENT`
#: (their character needed document context); phi entries are
#: :attr:`GlyphVerdictScope.FONT_PROGRAM` (the outline pins phi context-free).
_GLYPH_VERDICTS: tuple[GlyphVerdict, ...] = (
    GlyphRepair(
        # Table 1, p4 of the combustion-conditions paper this lane's M2 target lives
        # in. Its temperature header renders "T (<glyph 2> C)" where glyph 2 of the
        # /F7 font is the unmapped /C14.
        scope=GlyphVerdictScope.DOCUMENT,
        document_sha256="9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b",
        font_program_sha256="fef02938850ee399076ea5efc5960c08a91546241c2814ea86ab49011919bed1",
        font_base_name="NNEIFK+AdvP4C4E74",
        glyph_name="/C14",
        replacement="°",
        evidence=(
            "Read from the document itself, in order of authority: (1) the font's "
            "/Encoding /Differences binds code 2 to the non-semantic name /C14, and its "
            "/ToUnicode CMap is silent about it (codespace <bc><fe> only), so the "
            "document does not decode this glyph; (2) the embedded 633-byte CFF program "
            "(sha256 above) draws C14 as two concentric near-circular closed contours -- "
            "an annulus of outer bbox [56,443]x[56,444] in the 1000-unit em, advance "
            "width 500 -- i.e. a small ring; (3) the page typesets it at 70% of the "
            "surrounding body height (font_height 4.88 vs 6.97) with its baseline raised "
            "2.55 pt, a superscript placement. A small raised ring is the degree sign "
            "U+00B0; the ring-shaped alternatives are excluded by that same placement "
            "and proportion (U+2218 RING OPERATOR sits centred on the math axis, U+25CB "
            "WHITE CIRCLE is full-size). No neighbouring text was used to reach the "
            "conclusion; that the repaired header reads 'T (°C)' is corroboration, "
            "not evidence."
        ),
    ),
    GlyphRepair(
        # Same document, Table 1. Its equivalence-ratio column header, and the caption
        # ("Table 1 <glyph> Measurement conditions"), and the ranges "0.6<glyph>1.0"
        # etc., all draw an en-dash from the /F2 symbol font whose WinAnsi 'e' slot
        # holds it. The document decodes it to a literal 'e' -- valid Latin, so it
        # surfaces MAPPED and nothing flags it. 188 glyphs in this document.
        scope=GlyphVerdictScope.DOCUMENT,
        document_sha256="9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b",
        font_program_sha256="08ab6520b901000f5ccd4ff5cf535ccde7465ce0d985ff38d7258e3574e93f40",
        font_base_name="NNEIEF+AdvPS44A44B",
        glyph_name="e",
        replacement="–",
        evidence=(
            "Read from the document itself. (1) The font carries WinAnsiEncoding with "
            "NO /ToUnicode and NO /Encoding /Differences, so byte 0x65 decodes to 'e' "
            "purely by the base encoding -- the font asserts no Unicode of its own; its "
            "/Flags are 32 (Nonsymbolic), and even the embedded CFF names the glyph 'e'. "
            "Every name and flag AGREES with 'e'; only the outline disagrees. (2) The "
            "embedded 363-byte CFF program (sha256 above; charset ['.notdef','L','e'], a "
            "two-glyph symbol subset) draws 'e' as a SINGLE contour, bbox [50,700]x"
            "[274,326] in the 1000-unit em, advance width 750: a 650-unit horizontal bar "
            "52 units tall, floating on the math axis and never touching the baseline. A "
            "genuine 'e' (the body font's, for contrast) is a two-contour bowl on the "
            "baseline, bbox [47,509]x[-11,528]. A bar this wide excludes a hyphen (short, "
            "low); at 0.65 em it excludes an em-dash (~1 em). It is a dash-class mark; "
            "read as an en-dash U+2013 because this glyph is the document's range and "
            "caption separator (its math-axis sibling 'L', used once inside exp(-Ea/RT), "
            "is the minus -- separate entry). That the repaired caption reads "
            "'Table 1 - Measurement conditions' and the ranges read '0.6-1.0' is "
            "corroboration, not evidence."
        ),
    ),
    GlyphRepair(
        # Same document. The one occurrence of the /F2 symbol font's OTHER glyph, its
        # WinAnsi 'L' slot, drawn once inside a rate expression "[A T^n exp(<glyph>Ea/
        # RT)]" on p8. Decodes to a literal 'L'; a bar read as a capital L inside an
        # Arrhenius exponent is a silent sign that is not even a sign.
        scope=GlyphVerdictScope.DOCUMENT,
        document_sha256="9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b",
        font_program_sha256="08ab6520b901000f5ccd4ff5cf535ccde7465ce0d985ff38d7258e3574e93f40",
        font_base_name="NNEIEF+AdvPS44A44B",
        glyph_name="L",
        replacement="−",
        evidence=(
            "Same 363-byte CFF program as the en-dash entry above (charset "
            "['.notdef','L','e']); WinAnsi, no /ToUnicode, no /Differences, /Flags 32, "
            "CFF glyph name 'L' -- every name and flag agrees with 'L', only the outline "
            "disagrees. The 'L' glyph is a SINGLE contour, bbox [170,830]x[261,339], "
            "advance width unread here but the ink is a 660-unit horizontal bar 78 units "
            "tall centred on the math axis (y 261-339) -- structurally a dash/minus, not "
            "a letter L (which is a baseline-to-cap vertical plus a foot). Excludes "
            "hyphen (too wide) and em-dash (~1 em). Distinguished from its sibling 'e' "
            "en-dash by placement AND role: this glyph occurs once, between 'exp(' and "
            "'Ea/RT', i.e. as the operator of exp(-Ea/RT). A minus sign U+2212 (the math "
            "operator), not U+2013; the mathematical position is the one place the "
            "outline alone cannot choose between the two dash-class marks and the role "
            "decides."
        ),
    ),
    GlyphRepair(
        # Same document, Table 1 equivalence-ratio header cell (p4). The label phi is
        # drawn from the /F13 symbol font whose WinAnsi 'f' slot holds it; decodes to a
        # literal 'f'. Surfaces MAPPED. FONT_PROGRAM scope: the outline pins phi with no
        # document context, so the conclusion holds for any document embedding this program.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="45b1e0bf5d9e3a6e5e7af5f2b83ba95e1b19696d7eb4a0cea93dd412c80df3ac",
        font_base_name="NNEJBM+AdvPS4721B4",
        glyph_name="f",
        replacement="φ",
        evidence=(
            "Read from document 9c59f1c6...d298 (provenance; the conclusion is font-program "
            "scoped, not document scoped). (1) WinAnsiEncoding, no /ToUnicode, no /Differences; "
            "byte 0x66 decodes to 'f' by base encoding alone; /Flags 32 (Nonsymbolic); "
            "the embedded CFF even names the glyph 'f'. Every name and flag agrees with "
            "'f'; only the outline disagrees. (2) The embedded 533-byte CFF program "
            "(sha256 above; charset ['.notdef','f','g']) draws 'f' with THREE contours, "
            "bbox [44,564]x[-189,660] in the 1000-unit em, advance width 604: a bowl "
            "(outer + inner counter = two contours) crossed by a vertical stroke that "
            "DESCENDS to y=-189, well below the baseline. A genuine 'f' (body font, for "
            "contrast) is a SINGLE contour, bbox [47,436]x[0,777], with no descender at "
            "all. A circular bowl with a counter and a descending vertical stroke is a "
            "phi U+03C6; the deep descender excludes every Latin letter, and the closed "
            "bowl excludes the descenderless 'f' the byte decodes to. That the repaired "
            "header labels an equivalence-ratio column is corroboration, not evidence."
        ),
    ),
    GlyphRepair(
        # Same document. A SECOND symbol font (AdvPS3ECA66) whose 'f' slot also holds a
        # phi, drawn in the body text on pages 5-11 (11 occurrences), NOT in the table.
        # Registered so the document reports phi uniformly: a paper that renders phi
        # correctly in one place and as 'f' in another is worse than one uniformly
        # wrong. Same fault, distinct embedded program, distinct sha256.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="22a1e061856b2e27b6d4997553c4f76538889b56abdef7fa05ec049538e42605",
        font_base_name="NNFAME+AdvPS3ECA66",
        glyph_name="f",
        replacement="φ",
        evidence=(
            "Read from document 9c59f1c6...d298 (provenance; font-program scoped). "
            "WinAnsiEncoding, no /ToUnicode, no /Differences, "
            "byte 0x66 -> 'f' by base encoding; /Flags 32; CFF glyph name 'f'. The "
            "embedded 439-byte CFF program (sha256 above; charset ['.notdef','f'], a "
            "one-glyph symbol subset) draws 'f' with THREE contours, bbox [44,624]x"
            "[-189,664], advance width 666: the same bowl-plus-descending-stroke as the "
            "/F13 phi above, descender to y=-189. Contrast the body 'f': one contour, no "
            "descender. A phi U+03C6, on the same outline grounds. Used on pages 5-11 as "
            "the equivalence-ratio symbol in running text."
        ),
    ),
    # ---- Target document (9c59...): the impostors the outline does NOT pin. Each was
    # refused by an earlier reviewer because a plausible resemblance is not the
    # exclude-the-alternatives standard the phi repairs above meet; refusing turns a silent
    # wrong character into an honest UNRESOLVED_IMPOSTOR. FONT_PROGRAM scope: the refusal is a
    # font-program claim (this program's glyph is an impostor) independent of document.
    GlyphRefusal(
        # /F? symbol font (AdvPS3FDD77), WinAnsi 'w' slot, drawn x3 in the target.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="a77cf75d3face992f0f8db90afa59c9f63aa2dec2eff86f02fa39fe3ff247159",
        font_base_name="NNEKEG+AdvPS3FDD77",
        glyph_name="w",
        reason="a single wide horizontal bar of the tilde/dash class; the outline does not pin WHICH mark",
        evidence=(
            "Read from document 9c59f1c6...d298 (provenance; font-program scoped). "
            "WinAnsiEncoding, no /ToUnicode, no /Differences, /Flags 32; byte 0x77 -> 'w' by "
            "base encoding; the embedded 416-byte CFF (charset ['.notdef','bracketleft','w']) "
            "draws 'w' as a SINGLE contour, bbox [170,830]x[194,406] in the 1000-unit em, "
            "advance 1000: a 660-unit-wide horizontal mark 212 units tall floating on the math "
            "axis, rendered as a shallow wave (a tilde-shaped stroke). This is a dash/tilde "
            "class mark, not a letter -- but the outline alone cannot choose between a tilde "
            "U+223C, an approx/similar sign, a swung dash and a wide rule, and no document role "
            "excludes them. Refused, not guessed."
        ),
    ),
    GlyphRefusal(
        # Same program (AdvPS3FDD77), WinAnsi '[' slot. Present in the subset; a double bar.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="a77cf75d3face992f0f8db90afa59c9f63aa2dec2eff86f02fa39fe3ff247159",
        font_base_name="NNEKEG+AdvPS3FDD77",
        glyph_name="[",
        reason="two stacked horizontal bars of the equals/double-rule class; the outline does not pin WHICH",
        evidence=(
            "Read from document 9c59f1c6...d298 (provenance; font-program scoped). Same 416-byte "
            "CFF program as the tilde 'w' above (charset ['.notdef','bracketleft','w']); "
            "WinAnsiEncoding, no /ToUnicode, no /Differences, /Flags 32; byte 0x5B -> '[' by "
            "base encoding. The 'bracketleft' glyph is TWO contours, bbox [170,830]x[157,443], "
            "advance 1000: two parallel horizontal bars (upper [170,830]x[365,443], lower "
            "[170,830]x[157,235]), each a 660-unit rule. A double-bar mark -- most likely an "
            "equals U+003D, but the outline does not exclude an identical-to U+2261 fragment, a "
            "parallel-to U+2225, or a double rule, and there is no role to decide. Refused."
        ),
    ),
    GlyphRefusal(
        # AdvPS4721B4 (same program as the target phi 'f'), WinAnsi 'g' slot, drawn x3.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="45b1e0bf5d9e3a6e5e7af5f2b83ba95e1b19696d7eb4a0cea93dd412c80df3ac",
        font_base_name="NNEJBM+AdvPS4721B4",
        glyph_name="g",
        reason="a single-contour deep descender resembling a gamma, but not excluded from its alternatives",
        evidence=(
            "Read from document 9c59f1c6...d298 (provenance; font-program scoped). Same program "
            "as the phi 'f' repair above (charset ['.notdef','f','g']); WinAnsiEncoding, no "
            "/ToUnicode, no /Differences, /Flags 32; byte 0x67 -> 'g' by base encoding. The 'g' "
            "glyph is a SINGLE contour, bbox [40,549]x[-190,447], advance 593: a stroke that "
            "descends to y=-190 with no closed counter, resembling a lowercase gamma. But a "
            "single open descender is exactly what the phi repair required MORE than to convict: "
            "the phi is pinned by a bowl-plus-through-stroke structure that excludes its "
            "alternatives, and this glyph has no comparable feature excluding an eta, a script "
            "descender or an integral-like mark. Held to the same standard, it refuses."
        ),
    ),
    # ---- 2012 document (7cc5...): en-dash repair; tilde and RHO refusals.
    GlyphRepair(
        # AdvPS44A44B 'e' slot: the en-dash, byte-identical outline to the target's. Drawn x90.
        scope=GlyphVerdictScope.DOCUMENT,
        document_sha256="7cc544150b26056b6bfcc48eec248943c8bbd72de99c469c6d7d50c672ed564e",
        font_program_sha256="6ace515f455d19c37d4f715242c1c2e5a4d049853bdc6ad2bce5bbf0dee5654b",
        font_base_name="JIAKBG+AdvPS44A44B",
        glyph_name="e",
        replacement="–",
        evidence=(
            "Read from this document. WinAnsiEncoding, no /ToUnicode, no /Differences, /Flags 32; "
            "byte 0x65 -> 'e' by base encoding; the embedded 375-byte CFF (charset "
            "['.notdef','dollar','e']) draws 'e' as a SINGLE contour, bbox [50,700]x[274,326], "
            "advance 750 -- a 650-unit horizontal bar 52 units tall on the math axis, the "
            "byte-for-byte outline of the target document's confirmed en-dash 'e'. A bar this "
            "wide excludes a hyphen and at 0.65 em an em-dash; read as en-dash U+2013 because "
            "this glyph is the document's range/caption separator (drawn 90 times). Dash class, "
            "so DOCUMENT scope: the en-dash-vs-minus choice used the separator role. That the "
            "repaired ranges read '0.6-1.0' is corroboration, not evidence."
        ),
    ),
    GlyphRefusal(
        # AdvPS3FDD77 'w' slot: the same tilde outline as the target's, distinct program. x1.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="6badb7b4d560bde0581a37225a626f5f12b76015e627625528f2ab190832452a",
        font_base_name="JIAKIH+AdvPS3FDD77",
        glyph_name="w",
        reason="a single wide horizontal bar of the tilde/dash class; the outline does not pin WHICH mark",
        evidence=(
            "Read from document 7cc54415...d564 (provenance; font-program scoped). WinAnsiEncoding, "
            "no /ToUnicode, no /Differences, /Flags 32; byte 0x77 -> 'w'; the embedded 416-byte "
            "CFF (charset ['.notdef','bracketleft','w']) draws 'w' as a SINGLE contour, bbox "
            "[170,830]x[194,406], advance 1000 -- the same tilde-class bar as the target's "
            "AdvPS3FDD77 'w'. Not pinned to a specific tilde/rule/dash; refused for the same "
            "reason as that one."
        ),
    ),
    GlyphRefusal(
        # AdvPS4721B4 'r' slot, drawn x5. The census labelled this "phi", but the OUTLINE is a
        # RHO -- one counter, a bowl topping out at x-height, a left descender, and no
        # through-stroke ascending to cap height. Both confirmed corpus phis (target 'f',
        # 2014 'f') have a through-stroke to ~660 and TWO counters; this has neither. The
        # brief's own "~54 phi" total (target 17 + 2014 37) excludes these 5. Refused rather
        # than repaired to phi (that would be repairing against the outline on a label) or to
        # rho (a repair the operator never confirmed).
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="6bfd9d5860c56e9e6585c75e772c8927b9d2cbb3ff7f6ae9bc58f8a5fd5c1402",
        font_base_name="JIAKAD+AdvPS4721B4",
        glyph_name="r",
        reason=(
            "a closed bowl with a left descender -- a rho, NOT the phi the census labelled it; "
            "refused pending confirmation"
        ),
        evidence=(
            "Read from document 7cc54415...d564 (provenance; font-program scoped). WinAnsiEncoding, "
            "no /ToUnicode, no /Differences, /Flags 32; byte 0x72 -> 'r'; the embedded 1045-byte "
            "CFF (an 8-glyph symbol subset ['.notdef','F','h','k','l','r','s','u']) draws 'r' as "
            "TWO contours, bbox [18,469]x[-189,443], advance 500: an outer bowl whose stem "
            "descends to y=-189 on the left (x 18-93), plus ONE inner counter [183,389]x[17,413]. "
            "It tops out at x-height (443) with NO stroke ascending above the bowl. The two "
            "confirmed phi glyphs in this corpus (target/2014 'f') each have a vertical stroke "
            "spanning descender to cap height (~660) and TWO counters split by it; this glyph has "
            "one counter and no ascending stroke, so it is NOT that phi -- structurally it is a "
            "rho U+03C1 (bowl + left descender). Rendered, it is unmistakably a rho. The census "
            "annotation 'phi' is contradicted by the outline (and by the brief's own ~54-phi "
            "total, which excludes these 5). Refused: repairing to phi would be a label over the "
            "outline, and rho is a conclusion the outline supports but no reviewer has confirmed."
        ),
    ),
    # ---- 2014 document (a083...): en-dash and phi repairs; beta refusals (x2 programs).
    GlyphRepair(
        # AdvPS44A44B 'e' slot: the en-dash again, byte-identical outline. Drawn x153.
        scope=GlyphVerdictScope.DOCUMENT,
        document_sha256="a08397afba890749545aa69ea54769e002d40bbc78016eeabe0bb48d2120f73c",
        font_program_sha256="902c8300b657c496935a839491f176ee51b3263b8de9e841dafd7a5071441244",
        font_base_name="FCHIBE+AdvPS44A44B",
        glyph_name="e",
        replacement="–",
        evidence=(
            "Read from this document. WinAnsiEncoding, no /ToUnicode, no /Differences, /Flags 32; "
            "byte 0x65 -> 'e'; the embedded 347-byte CFF (charset ['.notdef','e']) draws 'e' as a "
            "SINGLE contour, bbox [50,700]x[274,326], advance 750 -- the byte-for-byte en-dash "
            "outline of the target and 2012 documents. Read as en-dash U+2013, the document's "
            "range/caption separator, drawn 153 times. Dash class -> DOCUMENT scope."
        ),
    ),
    GlyphRepair(
        # AdvP4721B4 'f' slot: phi, the bowl-plus-through-stroke structure. Drawn x37.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="e0afd5cb18fb6e556c5af5544747abb85212d52cd2a0365ae8eb4229b662bb76",
        font_base_name="FCHIHO+AdvP4721B4",
        glyph_name="f",
        replacement="φ",
        evidence=(
            "Read from document a08397af...0f73c (provenance; font-program scoped). WinAnsiEncoding, "
            "no /ToUnicode, no /Differences, /Flags 32; byte 0x66 -> 'f'; the embedded 686-byte CFF "
            "(charset ['.notdef','a','b','f']) draws 'f' with THREE contours, bbox "
            "[44,564]x[-189,660], advance 604: a vertical stroke spanning the descender (y=-189) to "
            "cap height (y=660), crossing a bowl split into TWO counters ([290,484]x[17,416] and "
            "[124,328]x[15,417]) on either side of it. A bowl crossed by a through-stroke that "
            "extends above and below is a phi U+03C6; the through-stroke and paired counters "
            "exclude the descenderless body 'f' the byte decodes to. Same structure as the target "
            "document's confirmed phi. Rendered, unmistakably a phi. Used x37 as the equivalence "
            "ratio."
        ),
    ),
    GlyphRefusal(
        # AdvP4721B4 'b' slot (same program as the 2014 phi 'f'), drawn x14. A beta.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="e0afd5cb18fb6e556c5af5544747abb85212d52cd2a0365ae8eb4229b662bb76",
        font_base_name="FCHIHO+AdvP4721B4",
        glyph_name="b",
        reason=(
            "an ascender-plus-double-bowl resembling a beta, but not the confirmed en-dash/phi "
            "fault; refused pending confirmation"
        ),
        evidence=(
            "Read from document a08397af...0f73c (provenance; font-program scoped). WinAnsiEncoding, "
            "no /ToUnicode, no /Differences, /Flags 32; byte 0x62 -> 'b'; same 686-byte CFF as the "
            "phi 'f' above (charset ['.notdef','a','b','f']) draws 'b' with TWO contours, bbox "
            "[35,522]x[-174,693], advance 552: a form that BOTH ascends (to y=693) and descends "
            "(to y=-174), unlike a Latin 'b' which ascends but never descends. Rendered, it is a "
            "beta (a left stem carrying two stacked bowls). The census flagged it with a question "
            "mark, and the brief is explicit that a question mark is not evidence: repairing it to "
            "beta while holding gamma/rho/the bars to the exclude-the-alternatives standard would "
            "apply a weaker bar to the one that merely LOOKS plausible -- the phi-as-f fault run "
            "backwards. Refused: the honest state for a glyph looked at and not confirmed."
        ),
    ),
    GlyphRefusal(
        # AdvP3ECA66 'b' slot, a SECOND program, drawn x1. Same beta fault, distinct program.
        scope=GlyphVerdictScope.FONT_PROGRAM,
        font_program_sha256="06fcebb4ccfa8272a362e3f81be7ef5dfac5527d40e1fc5b6b74d56bc21e0ab4",
        font_base_name="FCHNAI+AdvP3ECA66",
        glyph_name="b",
        reason=(
            "an ascender-plus-double-bowl resembling a beta, but not the confirmed en-dash/phi "
            "fault; refused pending confirmation"
        ),
        evidence=(
            "Read from document a08397af...0f73c (provenance; font-program scoped). WinAnsiEncoding, "
            "no /ToUnicode, no /Differences, /Flags 32; byte 0x62 -> 'b'; the embedded 553-byte CFF "
            "(charset ['.notdef','a','b']) draws 'b' with TWO contours, bbox [35,569]x[-177,697], "
            "advance 593: the same ascending-and-descending beta form as the AdvP4721B4 'b', in a "
            "distinct embedded program (drawn once). Refused on the same grounds -- registered so "
            "the document reports this glyph uniformly rather than refusing it in one program and "
            "reading it as 'b' in another."
        ),
    ),
)

_assert_no_overlapping_verdicts(_GLYPH_VERDICTS)


def registry_glyph_repairs_for_document(document_sha256: str) -> tuple[GlyphRepair, ...]:
    """The DOCUMENT-scoped glyph REPAIRS the table lane holds in force for this document.

    This is the fragment lane telling the text lane what it repairs -- the seam where the
    two lanes meet. The table lane runs every fragment of a PDF through :data:`_GLYPH_VERDICTS`
    (:func:`extract_fragments`); the running-text lane (``carmel.agents.tools.extract`` ->
    pypdf ``extract_text``) never traverses that path, so its stored characters keep the raw
    mis-decode. A char-span grounding into that text (``carmel.services.dataset_producer``)
    consults THIS function so it can name a mis-decode instead of reporting an
    uninformative "not found" (see ``ground_quote``'s ``repairs`` argument).

    Returns ONLY :class:`GlyphRepair` entries, and only :attr:`GlyphVerdictScope.DOCUMENT`-scoped
    ones whose ``document_sha256`` matches. Two deliberate exclusions:

    * :class:`GlyphRefusal` entries are omitted: a refusal leaves the impostor's WRONG Latin
      decode in place (``'e'``, ``'['``), which collides with the same character legitimately
      elsewhere in the text -- the running-text lane keeps no per-glyph font-program identity,
      so those cannot be located without re-aligning the two lanes, and refusing every ``'e'``
      would reject honest text. The impostor direction is therefore a design finding, not a
      thing this seam can close (see the ticket's Intent and this module's ``GlyphMapping``).
    * :attr:`GlyphVerdictScope.FONT_PROGRAM`-scoped verdicts are omitted: whether one is in
      force depends on the embedded program being PRESENT, which a sha-only query cannot
      confirm without parsing the document, and the same character (a genuine ``φ``, a genuine
      U+2212 minus) can be correctly decoded elsewhere in the very same file. Including one
      unconfirmed would refuse a real glyph. The text-lane mis-decodes this closes -- the
      en-dash, the degree sign, the exponent minus -- are all DOCUMENT-scoped by construction
      (their character needed document context; see :data:`_GLYPH_VERDICTS`).

    The result never widens the registry's evidence standard (a non-goal); it only re-reads,
    keyed on the document, repairs the registry already asserts.
    """
    return tuple(
        verdict
        for verdict in _GLYPH_VERDICTS
        if isinstance(verdict, GlyphRepair)
        and verdict.scope is GlyphVerdictScope.DOCUMENT
        and verdict.document_sha256 == document_sha256
    )


#: Ceiling on the decompressed size of one embedded font program before its sha256 is
#: taken. An embedded CFF/TrueType/Type1 subset is small -- the corpus's symbol subsets
#: decompress to well under 2 kB, and even a full non-subset font is a few hundred kB --
#: so this generous 6 MB bound never refuses a real program while still capping the
#: inflate below.
MAX_FONT_PROGRAM_BYTES = 6_000_000

MAX_FONT_PROGRAM_BYTES_PER_DOCUMENT = 60_000_000
"""Cumulative decompressed font-program bytes one document may inflate for verdict scoping.

:data:`MAX_FONT_PROGRAM_BYTES` bounds ONE stream; this bounds their SUM across every distinct
embedded font on every page, the same way :data:`MAX_PDF_GLYPH_INTERVALS` bounds a document
and not a page. The per-stream ceiling alone is not a document bound: a crafted PDF embedding
many distinct sub-6 MB font programs, or repeating them page by page, forces one bounded
inflate each and their total is unbounded. The sibling page-content path already carries a
document-wide cap (``glyph_budget`` is "the document-wide cap minus what earlier pages already
used"); this makes the font path follow the same pattern.

Headroom, not a corpus measurement: real papers embed SUBSETTED symbol fonts totalling far
under a megabyte, so this 10x-the-per-stream ceiling never bites a genuine document. It exists
so the cumulative inflate of a many-font bomb is bounded rather than only each font individually.
When it IS reached the last fonts fail closed -- their sha comes back ``None``, no verdict
matches, and their repairs are simply not applied -- which is the refusing direction."""


def _font_program_sha256(font: Any, limit: int = MAX_FONT_PROGRAM_BYTES) -> tuple[str | None, int]:
    """The sha of this font's embedded program and the bytes it inflated, or ``(None, 0)``.

    Returns ``(sha256_hex, decompressed_length)`` on success and ``(None, 0)`` -- no verdict
    can match, nothing was inflated -- for a font with no embedded program at all, and for
    every failure while reaching or inflating one: fail closed, a verdict that cannot verify
    its scope does not apply. The length is what the caller charges against the document-wide
    font budget, so ``0`` on the no-match paths is exact: those paths inflate nothing.

    ``get_data()`` inflates the font stream, and under FONT_PROGRAM-scoped verdicts this
    runs on the fonts of ANY supplied PDF, not only registered documents: a context-free
    verdict matches every document embedding its program, so the narrowing that used to
    guarantee "these bytes are already pinned in full" no longer holds. The inflate is
    therefore BOUNDED here exactly as the content lane bounds its own -- a
    :func:`_decoded_content_length` pass with ``limit`` as the ceiling runs FIRST and refuses
    (the ``except`` below turns it into ``(None, 0)``, i.e. no match) the moment the stream
    would decompress past the bound, so ``get_data`` only ever materialises bytes that pass.
    ``limit`` is the smaller of :data:`MAX_FONT_PROGRAM_BYTES` (this one stream) and the
    document's remaining font budget, so a document that has already spent its budget passes a
    non-positive ``limit`` and every further font fails closed. Measured on the corpus: every
    registered program is a single ``/FlateDecode`` (or unfiltered) stream whose bounded length
    equals ``get_data``'s exactly, so the bound costs nothing and changes no repair.
    """
    try:
        stream = font.font_descriptor.font_file
        if stream is None:
            return None, 0
        length = _decoded_content_length(stream, limit)
        return hashlib.sha256(bytes(stream.get_data())).hexdigest(), length
    except Exception:  # noqa: BLE001 - any failure to identify the program (incl. an over-bound inflate) is "no match"
        logger.debug("embedded font program could not be identified for verdict scoping", exc_info=True)
        return None, 0


def _text_pieces(show: Any) -> list[str] | None:
    """This show's PUBLISHED text, partitioned one piece per glyph code, or ``None``.

    The glyph-aligned view of ``show.text``: :func:`_glyph_substrings` partitions the
    DECODED value per code, and this maps each substring through the font's
    ``character_map`` exactly the way pypdf built the text -- so the pieces
    concatenate to ``show.text`` by construction, and that construction is verified
    rather than trusted: any disagreement (a pypdf decode this partition no longer
    mirrors) returns ``None``, and every per-glyph consumer -- the repair table, the
    glyph/ink evidence -- degrades to "no per-glyph view" rather than operating on an
    alignment that is a guess.

    This partition is the one sound way to slice fragment text at glyph boundaries.
    Glyph count can differ from ``len(text)`` (a ``/Differences`` name is one code and
    four characters), so any consumer slicing text by glyph COUNT is already wrong;
    slicing at piece boundaries is exact.
    """
    substrings = _glyph_substrings(show)
    if substrings is None:
        return None
    if isinstance(show.value, bytes):
        character_map = show.font.character_map
        pieces = ["".join(character_map.get(c, c) for c in substring) for substring in substrings]
    else:
        pieces = substrings
    if "".join(pieces) != show.text:
        return None
    return pieces


def _judged_pieces(pieces: list[str], verdicts: dict[str, GlyphVerdict]) -> tuple[list[str], GlyphMapping] | None:
    """Apply the font-program-scoped verdicts to one glyph-partitioned fragment.

    ``verdicts`` maps glyph_name -> the :class:`GlyphVerdict` in force for this show's font
    program in this document (the caller has narrowed by both). Returns the pieces to keep
    and the resulting :class:`GlyphMapping`, or ``None`` when nothing changed and the caller
    should leave the fragment exactly as it found it.

    The match is per GLYPH, never string substitution on the joined text, and that is the
    whole safety argument. A piece is exactly one glyph's decode (:func:`_text_pieces`), so a
    key that is a marker (``/C14``) matches only a glyph that decoded to that marker, and a
    key that is a plain character (``f``) matches only a glyph that decoded to that one
    character -- ordinary text that merely SPELLS ``f`` or ``/C14`` arrives as several pieces
    and cannot match. There is deliberately no "only a marker is eligible" gate: the
    mis-decoded glyphs the registry also covers (a symbol font's ``f`` that is really a phi)
    are valid Latin, never markers, and the font-program scope -- not the shape of the piece
    -- is what bounds the match.

    Precedence and the no-change cases:

    * A :class:`GlyphRefusal` on ANY piece makes the WHOLE fragment
      :attr:`~GlyphMapping.UNRESOLVED_IMPOSTOR`, text UNMODIFIED. An undecided impostor is not
      shipped as data even beside a repairable neighbour: a partly-repaired-partly-impostor
      string would read as clean while carrying a glyph Carmel refused to name.
    * Otherwise every match is a :class:`GlyphRepair` and the pieces are replaced. If a marker
      survives every repair the fragment stays UNMAPPED (returns ``None`` -- text unmodified,
      the published contract for a half-repaired fragment); else it is
      :attr:`~GlyphMapping.REPAIRED`.
    * No verdict matched any piece: returns ``None``, and the fragment keeps the mapping it
      already had (MAPPED or UNMAPPED).
    """
    matched = False
    refused = False
    repaired: list[str] = []
    for piece in pieces:
        verdict = verdicts.get(piece)
        if isinstance(verdict, GlyphRepair):
            matched = True
            repaired.append(verdict.replacement)
        elif isinstance(verdict, GlyphRefusal):
            matched = True
            refused = True
            repaired.append(piece)
        else:
            repaired.append(piece)
    if not matched:
        return None
    if refused:
        return list(pieces), GlyphMapping.UNRESOLVED_IMPOSTOR
    if _UNMAPPED_MARKER_RE.search("".join(repaired)):
        return None
    return repaired, GlyphMapping.REPAIRED


@dataclass(frozen=True)
class TextFragment:
    """One text-show operation, with where on the page it drew.

    A fragment is NOT a word and NOT a cell. Real publisher PDFs emit roughly 2.3
    fragments per word: a single word arrives in several pieces, consecutive
    fragments can OVERLAP in x (kerning), and bare single-space fragments interleave
    with them. Any caller that assumes one fragment is one token is already wrong.
    """

    page: int
    """1-indexed page number, counted AFTER phantom page-tree entries are dropped, so
    that it agrees with the numbering ``carmel.agents.tools.extract`` already
    produces. See :func:`extract_fragments` for why that agreement is not optional."""

    text: str
    """The decoded text as the document emitted it, EXCEPT where the glyph verdict registry
    rewrote it. A :class:`GlyphRepair` in force for this show's font program replaces the
    offending glyph pieces and :attr:`glyph_mapping` is then :attr:`~GlyphMapping.REPAIRED`; a
    :class:`GlyphRefusal` leaves the text byte-for-byte as emitted but marks
    :attr:`glyph_mapping` :attr:`~GlyphMapping.UNRESOLVED_IMPOSTOR`, so a downstream consumer
    refuses the mis-decoded character rather than trusting it. Read :attr:`glyph_mapping` to
    know which of the three this is; it is the only field that says whether the text was
    touched, and treating this string as the raw document emission is exactly the wrong trust
    assumption on a repaired fragment."""

    x_start: float
    """Absolute page-space x of the first glyph."""

    x_end: float
    """Absolute page-space x after the last glyph. Note ``x_end`` of one fragment may
    exceed ``x_start`` of the next: kerned runs overlap."""

    baseline_y: float
    """Absolute page-space y of the baseline. This is the BASELINE, not a bounding-box
    edge -- comparing it against a bbox-based engine leaves a constant descender
    offset."""

    font_height: float
    """The RENDERED height of the text, in page-space units -- the nominal size already
    composed with the text matrix.

    Deliberately NOT pypdf's ``font_size``, which is the raw ``Tf`` operand and is a
    trap. Publishers overwhelmingly emit ``Tf /F1 1`` and carry the real size in the
    text matrix instead, so ``font_size`` is **1.0 for 78 169 of the 78 178 fragments**
    in the real corpus: a field that looks like a point size, reads as a point size, and
    is a constant. Any font-relative policy built on it -- a vertical band of
    ``0.6 * font_size``, say -- silently degenerates to a fixed 0.6 units without ever
    looking wrong. ``font_height`` on the same shows recovers the actual 7.97 / 6.38 /
    9.0 pt type. Recording the composed height is the only honest choice; recording the
    operand and calling it a size is how a downstream threshold becomes a constant."""

    rotated: bool
    """True when the text is rotated with respect to the page. Retained rather than
    dropped; see :func:`_page_fragments`."""

    glyph_mapping: GlyphMapping

    ink_x_end: float | None = None
    """Absolute page-space x where the last glyph's own advance width ends -- the
    fragment's INK extent, as opposed to :attr:`x_end`, its ADVANCE extent.

    The two differ because the PDF text-space displacement charges character spacing
    (and word spacing, on a space) after EVERY glyph including the last (ISO 32000-1
    9.4.4), so ``x_end`` reaches one full character space past anything the fragment
    drew. On body text that trailing charge is zero and the two are equal; on a
    ``Tc``-spaced run it is not, and the gap is not small: the real corpus table this
    was built against carries a two-glyph show whose advance extent is 191.736 pt while
    its ink ends 92.019 pt earlier. A containment test that reads ``x_end`` refuses
    that fragment for spacing it never drew.

    Both extents are published because they answer different questions and each has
    call sites that need it. ``x_end`` is where the NEXT show starts, the fail-closed
    reading for anything that must not undercount a fragment's reach (the clip guard,
    ``pdf_cells``' neighbour geometry); this one is what the fragment actually
    occupies, the honest reading for containment and cut tests in
    :mod:`carmel.services.pdf_tables`.

    This is the extent of the last glyph's WIDTH, not a drawn-pixels bound: a glyph
    box is still the em-square estimate it always was, internal spacing is still
    inside the span, and a trailing SPACE glyph's width counts as ink here (its width
    is what the document reserves for it; erring wide costs a refusal, not a
    publication).

    ``None`` when the per-glyph decomposition needed to charge the trailing spacing is
    unavailable -- a dict-encoded show with an undecodable byte, or a fragment built
    outside :func:`extract_fragments` (every synthetic test fixture). Consumers fall
    back to :attr:`x_end`, which restores exactly the pre-field behaviour and fails in
    the refusing direction, because the advance extent is never narrower than the ink."""

    glyph_intervals: tuple[tuple[str, float, float], ...] | None = None
    """Per-glyph evidence: one ``(text piece, ink x-start, ink x-end)`` per glyph code,
    in drawing order, in absolute page space -- or ``None`` when not recorded.

    This is EVIDENCE, not a decision. A single show operator can draw the last glyph of
    one table cell and the first glyph of the next, carrying the inter-column gap as
    character spacing -- the real target's ``(91)Tj`` with ``13.1949 Tc`` draws ``9``
    closing one column's value and ``1`` opening the next's, 92 pt apart. Extraction
    cannot split that safely: it has no footprint, no rows and no table context, and
    the per-glyph spacing it would need is otherwise discarded during construction. So
    it records where each glyph's ink actually landed AFTER all spacing and
    displacement -- final page-space intervals, never raw spacing operands, because
    ``Tc`` is not the only gap source (``TJ`` displacements and ``Tw`` open internal
    space too) -- and the split decision lives one layer up, in
    :func:`carmel.services.pdf_tables.build_inventory`, where the columns exist to
    judge it against.

    The intervals are the WIDTH each glyph's advance reserves (``w0 * Tfs * Th``,
    through pypdf's own width lookup), excluding the ``Tc``/``Tw`` charged after it; a
    space glyph's width is an interval like any other. The text pieces are
    :func:`_text_pieces`' partition of :attr:`text` -- pieces concatenate to the text
    exactly, so a consumer can slice at piece boundaries where slicing by glyph count
    is unsound. Verified on the motivating fragment: the two glyphs land at
    [130.734, 134.583] and [226.602, 230.451], the second opening exactly where its
    column's other members sit.

    ``None`` for a rotated show, for a partition that is unavailable or no longer
    concatenates to the published text, for any non-finite or backwards interval, and
    for direct construction (synthetic fixtures) -- a strict subset of the situations
    where consumers already degrade, and never PARTIAL: a fragment carries its whole
    partition or none of it, and a consumer holding ``None`` must refuse to split
    rather than guess. Bounded document-wide by :data:`MAX_PDF_GLYPH_INTERVALS`,
    which truncates the extraction rather than shipping a fragment stripped of its
    evidence."""


@dataclass(frozen=True)
class FragmentPageFailure:
    """One page that could not be turned into fragments.

    Recorded rather than merely counted, mirroring
    :class:`carmel.agents.tools.extract.PageExtractionFailure`. A bare ``lossy=True``
    says "something was lost" without saying WHAT, so an operator cannot tell a
    single unreadable page from a document that mostly failed -- and a locator built
    from the pages that DID parse would look complete.
    """

    page: int
    error: str
    """Short, path-redacted description. Built by the text lane's own
    ``_describe_page_error`` so the redaction rules stay in one place."""


class FragmentAvailability(StrEnum):
    """Whether this lane produced anything, and if not, WHOSE FAULT that is.

    Four ways to have no fragments, and they are not one event. This was a single
    ``available: bool`` returned from three different sites, which is the
    UNVERIFIABLE-vs-FAILED conflation this codebase forbids, committed in production:
    an uninstalled pypdf, a refusing capability gate, and a document that defeated a
    healthy pinned engine all arrived as the same ``False``. The first two say nothing
    whatsoever about the document, because nothing ever looked at it; the third is a
    statement ABOUT the document.

    The members are named for the OWNER of the problem, because that is the axis the
    boolean destroyed and the one an operator needs -- and where ownership cannot
    honestly be established, the member says so rather than guessing (see
    ``READER_WALK_FAILED``). It is deliberately not an axis of severity: every
    non-``AVAILABLE`` member refuses exactly as much as every other, and nothing here
    is a licence to claim anything. See ``region_refusals``, which treats all four
    identically on purpose.
    """

    AVAILABLE = "available"
    """The pinned engine ran this document to completion. Says nothing about how MUCH
    was obtained -- ``lossy``, ``truncated`` and ``page_failures`` answer that, and an
    AVAILABLE extraction can still be badly incomplete."""

    ENGINE_ABSENT = "engine_absent"
    """pypdf is not installed, so nothing was read. Not an alarm: pypdf is an optional
    extra and CI's base job runs this way by design. Says NOTHING about the document,
    which is the whole reason it must not read like a document verdict.

    Reported ONLY for a ``ModuleNotFoundError`` naming ``pypdf`` itself. An installed
    pypdf whose own import fails -- a missing transitive dependency, a crash at import
    time, a package that no longer exports ``PdfReader`` -- is an alarm wearing the
    same ``ImportError`` coat, and reporting it as "the optional extra is not
    installed" would have been the original defect committed a second time. Those go
    to ``ENGINE_REFUSED``; the three shapes were measured, not assumed.

    ONE case is knowingly misreported and cannot be told apart here: a poisoned
    ``sys.modules["pypdf"] = None`` raises ``ModuleNotFoundError(name="pypdf")``, the
    same class and name as a genuine absence, because the interpreter itself treats it
    as absence. That is environment corruption wearing the supported configuration.
    Separating it means inspecting ``sys.modules`` before importing, which is a stranger
    thing for production code to do than the misclassification is worth -- but it IS a
    misclassification, and saying so is cheaper than a reader rediscovering it."""

    ENGINE_REFUSED = "engine_refused"
    """pypdf is installed and cannot be used: either :func:`_engine` rejected it at the
    front door -- moved internals, a changed ``TextStateParams`` shape or field order,
    a missing ``StreamObject._data``, a version that is unknown or not the pin -- or
    the package is present and broken enough that importing it failed. Nothing was
    read, so this too says NOTHING about the document. The deployment owns it.

    Which of those fired is NOT carried here; it is in a log line, and logs are
    ephemeral. Distinguishing them needs ``_engine`` to return a refusal rather than
    ``None``, which is a separate change. Until then an operator reading this state
    cannot tell "bump the pin" from "the internals moved" without the logs."""

    ENGINE_CONTRADICTED_GATE = "engine_contradicted_gate"
    """:func:`_engine` passed, and the engine then broke the same contract mid-walk
    (see :class:`_EngineMismatch`). A SEPARATE member from ``ENGINE_REFUSED`` rather
    than a sub-reason of it, because the owner differs: that one says fix your
    install, this one says the front-door gate is incomplete and Carmel must widen it.
    Collapsing the two would send someone to reinstall a correctly-installed pypdf."""

    READER_WALK_FAILED = "reader_walk_failed"
    """The pinned engine was in hand and reading this document raised for some other
    reason -- including in ``PdfReader``'s own CONSTRUCTION, which is inside the same
    guard, so "walk" here covers everything from opening the bytes onward and not only
    the per-page loop.

    Deliberately NOT called ``DOCUMENT_UNREADABLE``, which is what it is in the common
    case. That name would assert an ownership the ``except Exception`` cannot
    establish: the same clause catches ``MemoryError``, ``RecursionError``, a bug in
    this module, a failure inside ``_quiet_pypdf``'s enter/exit, and a failure while
    FORMATTING a page error. A ``MemoryError`` is also potentially transient, so even
    "re-running will not help" -- true of a corrupt document -- would be overclaiming.
    The name says what was observed: reading failed. ``KeyboardInterrupt`` and
    ``SystemExit`` derive from ``BaseException`` and are not caught at all, which is
    correct: an interrupt is not an extraction outcome."""


#: The states in which pypdf was actually reached and run, and therefore exactly the
#: states whose ``pypdf_version`` is set. Named rather than inlined because it is the
#: same fact in two places -- the invariant below and the return sites of
#: :func:`extract_fragments` -- and the two silently disagreeing is the shape of bug the
#: invariant exists to catch.
_ENGINE_RAN = frozenset(
    {
        FragmentAvailability.AVAILABLE,
        FragmentAvailability.ENGINE_CONTRADICTED_GATE,
        FragmentAvailability.READER_WALK_FAILED,
    }
)


@dataclass(frozen=True)
class FragmentExtraction:
    """The result of one whole-document extraction."""

    fragments: tuple[TextFragment, ...] = ()
    lossy: bool = False
    """True when this extraction is known to be incomplete: a page failed, a page
    could not be inspected, or the document was truncated. Mirrors
    ``ExtractedText.lossy``, and like it, fails toward admitting loss."""

    status: FragmentAvailability = FragmentAvailability.AVAILABLE
    """Whether this lane produced anything, and whose fault it is when it did not.

    Distinct from ``lossy``: anything but ``AVAILABLE`` means NOTHING here can be
    relied on and no claim about this document may be made, while ``lossy=True`` means
    what IS here is real but incomplete. Conflating them is the specific error this
    pair exists to prevent -- an engine-wide incompatibility that returned zero
    fragments while reporting itself available would read exactly like a legitimately
    empty document.

    ONE field rather than a boolean beside a reason. Two fields would be two facts
    where there is one, and would need an invariant enforced to keep them agreeing;
    a state that cannot be inconsistent needs no enforcement. ``available`` below is
    derived from it, so every existing consumer is untouched."""

    page_failures: tuple[FragmentPageFailure, ...] = ()
    truncated: bool = False
    """True when this document exceeded a bound and the rest was not processed.

    Covers BOTH bounds, exactly as ``ExtractedText.lossy`` covers both of the text
    lane's: more pages than ``MAX_PDF_PAGES``, or more fragments than
    :data:`MAX_PDF_FRAGMENTS`. The page cap is shared with the text lane so the two
    lanes agree on which pages exist; see :func:`extract_fragments`."""

    pypdf_version: str = ""
    """The pypdf version this extraction actually ran against.

    Recorded because the geometry is the evidence, and a pypdf that changed baseline
    semantics, CTM composition, page-rotation normalisation or ``TJ`` displacement
    could keep every attribute name intact -- passing :func:`_engine` -- while
    silently returning DIFFERENT numbers. No capability check can catch that, so the
    version travels with the result and the pin is asserted at runtime.

    Empty exactly when nothing was read: ``ENGINE_ABSENT`` and ``ENGINE_REFUSED``. It
    is therefore a PARTIAL discriminator between the states above, and must never be
    used as one -- it cannot separate those two from each other, it cannot separate
    ``ENGINE_CONTRADICTED_GATE`` from ``READER_WALK_FAILED``, and it discriminates at
    all only by accident. Setting ``version`` before the walk instead of after is an
    obviously-correct-looking edit that would silently erase what discrimination it
    does have, with no test failing. ``status`` is the carrier."""

    @property
    def available(self) -> bool:
        """True only for :attr:`FragmentAvailability.AVAILABLE`.

        Kept as a derived property, not a field, so it cannot disagree with
        ``status``. It is the coarse question -- "may I use any of this?" -- and it is
        the right question for a consumer that would treat all four failures the same
        way, which every consumer in the tree does today."""
        return self.status is FragmentAvailability.AVAILABLE

    def __post_init__(self) -> None:
        """Make the combinations this class's own prose forbids unconstructible.

        An internal consistency check on a construction path this module controls, not
        a security boundary -- ``pickle`` and ``object.__setattr__`` bypass it, and
        neither is a threat model here. Every rule below was written as a docstring
        first and enforced second, which is the wrong order: prose that the type does
        not enforce is a convention, and the suite had already grown a fixture that
        broke one of these while the docstring said it could not happen.

        ``status`` is type-checked because it is a :class:`StrEnum`: a raw
        ``"engine_absent"`` compares EQUAL to the member and is NOT it, so a consumer
        matching with ``is`` -- as this module tells them to -- would silently skip a
        state that reads correctly in every log line and repr.
        """
        if not isinstance(self.status, FragmentAvailability):
            raise TypeError(f"status must be a FragmentAvailability, not {type(self.status).__name__}")
        # `page_failures` and `truncated` are the two things that LOCATE loss, so either
        # of them present while the document-level flag says complete is a contradiction
        # inside one object.
        if (self.page_failures or self.truncated) and not self.lossy:
            raise ValueError("recorded page failures or truncation must set lossy")
        if self.status is not FragmentAvailability.AVAILABLE:
            if not self.lossy:
                raise ValueError(f"{self.status} must admit loss: an unavailable extraction is never complete")
            if self.fragments or self.page_failures or self.truncated:
                # "NOTHING here can be relied on" has to mean nothing is HERE. An
                # unavailable result carrying fragments invites a consumer to use them,
                # and one carrying page failures claims to know which pages failed in a
                # walk whose whole point is that it established nothing.
                raise ValueError(f"{self.status} carries evidence it cannot vouch for")
        if self.pypdf_version not in ("", _PINNED_PYPDF_VERSION):
            # Presence alone was not enough, and the gap was real: `pypdf_version="bogus"`
            # satisfied a boolean check while the field's entire meaning is "the PINNED
            # engine ran". Nothing else can legitimately appear -- `_engine` refuses every
            # other version before a walk can begin -- so any other value is a
            # construction error rather than a record of something that happened.
            raise ValueError(f"pypdf_version must be {_PINNED_PYPDF_VERSION!r} or empty, not {self.pypdf_version!r}")
        if bool(self.pypdf_version) is not (self.status in _ENGINE_RAN):
            # The version means "the engine that ran was this one". Recording it where
            # nothing ran claims an extraction happened; omitting it where something did
            # loses which engine to blame. Enforced for AVAILABLE too, at the cost of a
            # bare FragmentExtraction() no longer constructing -- which is correct: an
            # available extraction that never ran an engine is a fiction, and it was
            # only ever convenient as a fixture.
            raise ValueError(f"{self.status} must record the pypdf version exactly when the engine ran")


def _engine() -> tuple[Any, ...] | None:
    """Resolve pypdf's private layout-mode internals, or refuse.

    Everything this module needs lives under ``pypdf._text_extraction._layout_mode``:
    a leading-underscore package with no API-stability guarantee whatsoever. That is
    a deliberate, guarded trade. The alternative is re-implementing a content-stream
    interpreter -- operator dispatch for ``Tj``/``TJ``/``'``/``"``,
    ``Td``/``TD``/``T*``/``Tm``, ``Tc``/``Tw``/``Tz``/``TL``/``Ts``, ``cm``/``q``/``Q``
    nesting, AND font decoding through ``/Encoding`` and ``ToUnicode`` CMaps -- which
    is a far larger long-term liability than a pinned import, and would be a second
    decoder that could disagree with the one the shipped text lane already uses.

    So the dependency is taken, and the risk is handled where it actually bites: a
    pypdf upgrade that moves or changes these internals must make this module REFUSE
    loudly, never silently return different geometry. Returns ``None`` on any
    mismatch; the caller turns that into ``available=False``.
    """
    try:
        from pypdf._text_extraction._layout_mode._fixed_width_page import resolve_font
        from pypdf._text_extraction._layout_mode._text_state_params import (
            TextStateParams,
        )
        from pypdf.generic import ContentStream, StreamObject
    except Exception:  # pragma: no cover - exercised via monkeypatch in tests
        logger.debug("pypdf layout-mode internals unavailable", exc_info=True)
        return None

    # The imports resolving is not enough: the names could survive while the objects
    # behind them change shape. Check every piece actually read below -- including the
    # TextStateParams attributes, which must be checked HERE rather than only at the
    # point of use. A per-page AttributeError is caught as a page failure and degrades
    # to `lossy=True`, so an engine-wide mismatch would otherwise present as "a valid
    # document where every page happened to fail" instead of "the engine is wrong".
    #
    # `TextStateParams` is CONSTRUCTED here, not merely read, so the constructor's own
    # signature is part of the contract and gets its own check. Positional construction
    # is deliberate -- it is how pypdf's own `TextStateManager.text_state_params` builds
    # one -- and a release that reordered two same-typed float fields (`Tc` and `Tw`,
    # say) would keep every name, pass every `hasattr`, and silently swap character
    # spacing for word spacing in every advance this module computes.
    try:
        declared = tuple(field.name for field in dataclasses.fields(TextStateParams))
    except TypeError:  # only if pypdf stops using a dataclass
        logger.warning("pypdf TextStateParams is no longer a dataclass; fragments unavailable")
        return None
    if declared[: len(_REQUIRED_PARAM_FIELD_ORDER)] != _REQUIRED_PARAM_FIELD_ORDER:
        logger.warning(
            "pypdf TextStateParams fields are %s, not the expected %s; fragments unavailable",
            declared[: len(_REQUIRED_PARAM_FIELD_ORDER)],
            _REQUIRED_PARAM_FIELD_ORDER,
        )
        return None
    # Check FIELDS as well as class attributes. `TextStateParams` is a dataclass, and
    # a field without a default (`font_height` is one) exists only on instances, so a
    # bare `hasattr` on the class reports it missing and would refuse every healthy
    # pypdf. The properties (`tx`, `ty`, `text`, ...) do live on the class, so the
    # available surface is the union of the two.
    available_names = set(dir(TextStateParams))
    with contextlib.suppress(TypeError):  # only if pypdf stops using a dataclass
        available_names |= {field.name for field in dataclasses.fields(TextStateParams)}
    for attr in _REQUIRED_PARAM_ATTRS:
        if attr not in available_names:
            logger.warning("pypdf TextStateParams lacks %s; fragments unavailable", attr)
            return None

    # `_decoded_content_length` reads `StreamObject._data`, the RAW still-compressed
    # bytes, because bounding a decode needs the decode's INPUT and pypdf's public
    # `get_data()` returns only its output -- already materialised, which is the whole
    # defect. Probed on an INSTANCE and not on the class: pypdf sets `_data` in
    # `__init__` with no annotation and an empty `__slots__`, so the class carries no
    # trace of it and a `hasattr` there would refuse every healthy pypdf. Checked here
    # rather than at the point of use for the reason the whole gate exists: a per-page
    # AttributeError is caught as a page failure, so an engine-wide mismatch would
    # present as "a valid document where every page happened to fail".
    try:
        if not hasattr(StreamObject(), "_data"):
            raise AttributeError("_data")
    except Exception:
        logger.warning("pypdf StreamObject lacks _data; fragments unavailable")
        return None

    # The geometry is the evidence, and no attribute check can detect a release that
    # keeps every name while changing what the numbers MEAN. pypdf is pinned exactly
    # (`pypdf==6.14.2`) precisely because an extraction's dependency identity has to be
    # provable, and the pin's own comment in pyproject.toml makes bumping a deliberate
    # act with a re-extraction pass attached. Refusing here is the runtime half of that
    # policy: an unpinned pypdf makes this lane UNAVAILABLE rather than silently
    # differently-calibrated.
    try:
        installed = importlib.metadata.version("pypdf")
    except Exception:
        logger.warning("pypdf version is unknown; fragments unavailable")
        return None
    if installed != _PINNED_PYPDF_VERSION:
        logger.warning(
            "pypdf %s is not the pinned %s; fragment geometry is unverified, refusing",
            installed,
            _PINNED_PYPDF_VERSION,
        )
        return None
    return resolve_font, TextStateParams, ContentStream


_REQUIRED_PARAM_FIELD_ORDER = (
    "value",
    "font",
    "font_size",
    "Tc",
    "Tw",
    "Tz",
    "TL",
    "Ts",
    "transform",
)
"""The leading constructor parameters of ``TextStateParams``, in order.

:func:`_walk_operations` builds one per text-show operation POSITIONALLY, so this is a
signature contract and not a spelling check; see :func:`_engine` for what a silent
reordering would do.
"""

_REQUIRED_PARAM_ATTRS = (
    "text",
    "tx",
    "ty",
    "displaced_tx",
    "font_height",
    "rotated",
    # Read by `_advance` to compute the displacement one show applies to the text
    # matrix. pypdf's own `displacement_matrix()` wraps it, but this module needs the
    # scalar rather than the matrix, and needs it BEFORE the `Tc` correction is added.
    "word_tx",
    # Read by `_pen_x_after` to charge `Tc` once per glyph. Listed here with the rest
    # rather than probed at the point of use for the reason the docstring above gives:
    # a per-page `AttributeError` degrades one page to lossy, which would present an
    # engine-wide mismatch as "a valid document where every page happened to fail".
    "value",
    "_decoded_value",
    "font",
    "Tc",
    "Tz",
    "transform",
    # Read by `_ink_x_end` to charge the trailing word spacing when the final glyph is
    # a space. `Tc`/`Tz`/`transform`/`tx` were already listed for the sibling
    # corrections; `Tw` is the one the ink extent adds.
    "Tw",
)

_PINNED_PYPDF_VERSION = "6.14.2"
"""Must track the ``agents`` extra's exact pin in ``pyproject.toml``."""

#: Hard cap on how many fragments one document may yield, independent of
#: ``MAX_PDF_PAGES``. The page cap alone does NOT bound this lane: a single page may
#: carry unboundedly many text-show operations, and unlike the text lane -- whose
#: per-page output is one string it can measure against
#: ``MAX_EXTRACTED_TEXT_CHARS`` -- this one accumulates a Python object per operation.
#:
#: Sized from measurement, not taste. A fragment costs ~200 bytes (measured), the real
#: corpus runs 1071 fragments/page at 2.4 characters per fragment, so the text lane's
#: 500k-character ceiling corresponds to roughly 200k fragments for the largest
#: legitimate document (a supplementary-information PDF). 1M is 5x headroom over that
#: while bounding peak retention at ~200 MB -- the same order as the ~381 MB peak the
#: text lane's own cap was written against.
MAX_PDF_FRAGMENTS = 1_000_000

#: Hard cap on the total number of per-glyph interval entries one document may record,
#: counted across all pages, independent of both caps above -- because neither bounds
#: this: the fragment cap counts SHOWS and the page-content cap counts BYTES, and one
#: large show under the 6 MB content cap can carry millions of glyphs, each now costing
#: a retained ``(str, float, float)`` entry. Measured cost of the field at corpus scale
#: is ~45 bytes per entry (+12.6 MB over the design probe's 276,909 corpus entries); a
#: cap-saturating document would otherwise cost on the order of 160 MB for this field
#: alone.
#:
#: Sized against the corpus the way the sibling caps are: as shipped, the eight papers
#: record 274,212 entries in total and the largest single document 46,361 (the shipped
#: field skips rotated shows and the undecodable partition, hence slightly under the
#: probe's count), so two million is 43x headroom over the largest legitimate document
#: in hand while bounding the field's retention at ~91 MB -- inside the ~200 MB ceiling
#: the other two caps already express.
#:
#: Exhaustion TRUNCATES the extraction (``truncated=True, lossy=True``, same channel as
#: the fragment cap) rather than continuing without the field, deliberately: a fragment
#: stripped of its evidence would silently lose exactly the sub-fragment structure the
#: field exists to carry, and `build_inventory` refuses truncated documents wholesale,
#: which is the fail-closed direction.
MAX_PDF_GLYPH_INTERVALS = 2_000_000

#: Hard cap on the DECOMPRESSED content-stream bytes of a single page, checked before
#: pypdf parses it. A page over this is recorded as a page failure and skipped; the rest
#: of the document still extracts.
#:
#: This is the bound that :data:`MAX_PDF_FRAGMENTS` cannot provide. That cap counts
#: fragments, and both of the expensive things happen before the first one exists:
#: ``ContentStream`` materialises the whole operation list up front, and
#: ``recurse_to_target_op`` consumes one entire ``BT``/``ET`` group per call. The second
#: was assumed to be an edge case and is not -- measured on the corpus, **the largest
#: single ``BT`` group is a median 32% of its page's operations and up to 99.5%**, so on
#: real papers one group routinely IS the page and the between-groups budget check
#: cannot interrupt it. Bounding the page is therefore what bounds the group, and there
#: is no need to reach inside pypdf's parser to do it.
#:
#: Sized from measurement, and from the ceiling the sibling cap already declares. Parsed
#: operations cost **19.0 bytes of Python heap per decompressed byte at the median and
#: 33.2x at the worst** across 73 corpus pages (489 B per operation). 6 MB x 33.2 is
#: ~199 MB, the same ~200 MB peak retention :data:`MAX_PDF_FRAGMENTS` was sized against,
#: so the two caps now express one memory ceiling instead of two unrelated numbers.
#:
#: Headroom over legitimate documents is 7.2x: the largest real page in the 8-paper
#: corpus decompresses to 836,591 B (median 22,035 B). That corpus holds no
#: supplementary-information PDF, so a dense vector figure could plausibly exceed the
#: cap -- which costs ONE page, recorded as a failure and therefore visible, rather than
#: the document.
#:
#: **What it does not bound.** Not the transient decode -- that WAS true and is no longer,
#: and the correction is kept visible rather than quietly deleted because the superseded
#: text is the reason the current code is shaped as it is.
#:
#: This comment used to say a compression bomb still allocates its decompressed size once
#: before the check can reject it, on the grounds that bounding the decode would mean
#: reimplementing pypdf's filter stack. :func:`_decoded_content_length` now bounds it, and
#: the argument that said it could not be done was refuted by measuring rather than by
#: reasoning: every content stream in the corpus is a single-stage ``/FlateDecode`` with no
#: ``/DecodeParms``, so ``zlib.decompressobj`` bounds the only filter present and an exact
#: allowlist of one fails everything else closed. A bomb now stops at the cap.
#:
#: What the cap still does not bound, named in full because an earlier version of this
#: comment listed only the first of the two and so read as a completeness it did not have:
#:
#: 1. The COMPRESSED input. The stored bytes are read whole before any output limit
#:    applies, so a large incompressible stream under the cap is still copied and scanned.
#:    That is bounded by the artifact size instead (``max_artifact_bytes``, 25 MB), which is
#:    checked before this module ever sees the document.
#:
#: 2. **Every stream that is not** ``/Contents``. This cap is per page and reads one key;
#:    nothing else pypdf inflates passes through it. Two such paths exist and they are not
#:    equally exercised, so both are measured rather than asserted -- by instrumenting
#:    ``zlib`` and attributing every decompressed byte over the 8-paper corpus:
#:
#:    - FONT and ``/ToUnicode`` streams, inflated during pypdf's font resolution. PRESENT
#:      on 8 of 8 documents, at 1-6 kB over 2-13 calls each -- small, real, and outside
#:      this cap entirely.
#:    - ``/ObjStm``, which pypdf MUST inflate to resolve any object stored inside it, and
#:      which therefore runs before a page key can even be read. ZERO bytes on 8 of 8
#:      corpus documents, because these are all classic cross-reference-table PDFs. That
#:      zero is a property of the corpus and NOT a bound: object streams are ordinary in
#:      PDF 1.5+, and where one is present its inflation is unbounded.
#:
#: No bound is available for either IN THIS MODULE. Wrapping pypdf's filter stack is the
#: one thing it refuses to do (see :func:`_decoded_content_length`), and pypdf is pinned and
#: not to be patched or forked -- so within these functions the amplification on a
#: non-``/Contents`` stream is bounded by nothing, and the 25 MB artifact cap bounds its
#: INPUT rather than its output.
#:
#: "In this module" is doing real work in that sentence and an earlier draft omitted it,
#: which made it a stronger claim than the code can support. A bound DOES exist one layer
#: out: parse in a constrained worker process under ``RLIMIT_AS`` and a CPU timeout, kill it
#: on breach, and report the page or document as failed. That reaches every path at once --
#: ``/ObjStm``, fonts, the object graph, and pypdf's own allocations -- precisely because it
#: does not care which stream was responsible. It is unbuilt: ``extract_fragments`` runs
#: in-process today, and moving it is an architectural change with its own identity and
#: failure-reporting consequences, not a line in this file. Named here so the limit reads as
#: a decision with a known repair rather than as an impossibility.
MAX_PAGE_CONTENT_BYTES = 6_000_000

#: The ``error`` recorded for a page whose page-tree entry was UNINSPECTABLE.
#:
#: DUPLICATED from ``carmel.agents.tools.extract`` rather than imported from it, and the
#: duplication is deliberate. Hoisting the text lane's literal into a shared constant
#: changes ``extract_text``'s semantic-dependency closure, and that sha is the identity
#: under which every already-stored extraction was produced
#: (``tests/test_semantic_deps.py``). Perturbing a stored-evidence identity to
#: de-duplicate a string in a lane that has no stored artifacts yet is the wrong trade.
#: Drift is prevented instead by a test that reads the message the TEXT LANE ACTUALLY
#: EMITS at runtime and asserts this equals it -- a stronger check than a shared name,
#: because it compares behaviour rather than a symbol.
_UNINSPECTABLE_PAGE_ERROR = "page-tree entry could not be inspected; kept as a possible page"


#: The ONLY content-stream filter :func:`_decoded_content_length` will decode under a
#: size bound, as an exact single-stage chain rather than a member of one.
#:
#: An allowlist and not a blocklist, because the question is not "which filters are
#: dangerous" but "which can this module bound", and the answer has to shrink safely when
#: a filter nobody anticipated arrives. See :func:`_decoded_content_length` for the corpus
#: measurement of what refusing everything else costs (nothing, on 161 of 161 streams) and
#: for what that zero does not prove.
_ALLOWED_CONTENT_FILTER = "/FlateDecode"

#: The six bytes PDF counts as whitespace (ISO 32000-1 table 1). Spelled out rather than
#: reusing `bytes.isspace`, whose set is Python's and includes vertical tab, which PDF does
#: not; a guard that refuses on trailing bytes must use the FORMAT's definition of "not
#: content", or it decides what a PDF is allowed to contain on Python's authority.
_PDF_WHITESPACE = b"\x00\t\n\x0c\r "


class PageContentTooLarge(Exception):
    """One page's decompressed content stream exceeds :data:`MAX_PAGE_CONTENT_BYTES`.

    A page failure, not an engine failure: the class name lands verbatim in the stored
    ``FragmentPageFailure.error``, so it is spelled without a leading underscore.
    """


class PageContentUndecodable(Exception):
    """One page's content stream cannot be decoded under a size bound at all.

    Distinct from :class:`PageContentTooLarge`, and the distinction is the point: that one
    says the page is too big, this one says its SIZE COULD NOT BE ESTABLISHED. Conflating
    them would report a filter this module declines to handle as if the document were
    oversized, which is a claim about the document rather than about this module's reach.

    A page failure, not an engine failure, and spelled without a leading underscore for
    the same reason as its sibling: the class name lands verbatim in the stored
    ``FragmentPageFailure.error``. It is per-page on purpose -- one stream with an
    unexpected filter costs its own page and nothing else.
    """


class _EngineMismatch(Exception):
    """The pypdf engine is not shaped the way this module requires.

    Raised from page processing but deliberately NOT treated as a page failure: it
    says the ENGINE is wrong, not that one document page is. The distinction is the
    difference between ``available=False`` and a plausible-looking empty result.
    """


def _declared_filters(stream: Any) -> tuple[str, ...]:
    """The filter chain one stream declares, in application order.

    ``/Filter`` is legally a single name or an array of them, and the array form is a
    CHAIN: each stage feeds the next. Normalising both to a tuple is what lets the
    allowlist below be a comparison against one exact value rather than a membership
    test that would accept ``[/ASCII85Decode, /FlateDecode]`` because Flate is in it.
    """
    declared = stream.get("/Filter")
    if declared is None:
        return ()
    if isinstance(declared, list):
        return tuple(str(entry) for entry in declared)
    return (str(declared),)


def _decoded_content_length(contents: Any, limit: int) -> int:
    """Decompressed size of a page's ``/Contents``, bounded at ``limit`` bytes.

    ``/Contents`` is either one stream or an array of them that concatenate into a
    single stream, and only the sum bounds the parse -- a page split into a thousand
    small streams costs the same as one large one.

    Duck-typed on ``list`` rather than importing ``ArrayObject``, because pypdf's
    ``ArrayObject`` subclasses it and the engine tuple exists to keep the number of
    pypdf internals this module names to a minimum. A part that is neither raises, and
    the caller records the page as failed -- the fail-closed direction.

    **Why this does not simply call** ``get_data()``. It used to, and measuring a length
    that way requires materialising the whole decompressed stream first, so a
    compression bomb allocated its full size before :data:`MAX_PAGE_CONTENT_BYTES` could
    reject it -- the cap bounded the 33x parse amplification but not the decode. The
    bound is applied to the decode instead, by decompressing through
    :func:`zlib.decompressobj` with an output ceiling and refusing the moment input is
    left over.

    **Why an allowlist of exactly one filter is honest rather than a half-measure.** A
    guard that bounded Flate and quietly fell through to ``get_data()`` for anything else
    would read as a bound while being none, which is worse than the documented absence it
    replaced. This one fails CLOSED: any other filter, any chain, and any
    ``/DecodeParms`` is a page failure, recorded and visible. Measured before it was
    written -- across the 8-paper corpus, all 73 content-bearing pages and all 161
    streams are single-stage ``/FlateDecode`` with no ``/DecodeParms``, so the refusal
    costs nothing on real publisher articles. It has a price nonetheless, stated rather
    than glossed: **the refusal branch is unexercised by every document in hand**, so
    only synthetic fixtures reach it, and a legitimate PDF using a different filter loses
    a page. That is the fail-closed direction, and a page failure is recorded per page.

    **Why bare zlib is allowed to stand in for pypdf's Flate.** It is not a
    reimplementation of the filter stack, which is the thing this project refuses to do:
    it is the same ``zlib`` call pypdf makes first, and pypdf's extra machinery is
    RECOVERY for streams where that call fails. So whenever this succeeds, pypdf's decode
    of the same bytes is the same bytes -- verified on all 161 corpus streams, byte for
    byte, against ``get_data()`` as the oracle -- and whenever it fails, this refuses the
    page rather than guessing. The residual is a stream pypdf could recover and this
    cannot, which becomes a recorded page failure instead of a silent difference.

    ``ContentStream`` still decodes again immediately afterwards, and that second decode
    is now bounded by this one having passed: the duplicated CPU is accepted for the same
    reason as before, that the alternative is to let the parse run and measure the damage
    after it is done.
    """
    parts = contents if isinstance(contents, list) else [contents]
    total = 0
    for part in parts:
        stream = part.get_object()
        filters = _declared_filters(stream)
        if filters not in ((), (_ALLOWED_CONTENT_FILTER,)):
            raise PageContentUndecodable(
                f"page content stream declares filters {filters!r}; only a single "
                f"{_ALLOWED_CONTENT_FILTER} can be decoded under a size bound"
            )
        if stream.get("/DecodeParms"):
            raise PageContentUndecodable(
                "page content stream carries /DecodeParms; a predictor changes what a "
                "byte bound bounds, so the size cannot be established"
            )

        raw = bytes(stream._data)
        if not filters:
            # Unfiltered: the stored bytes ARE the content, so its size is already known
            # without decoding anything. Admitted rather than refused because there is
            # nothing here to bound -- refusing it would be refusing the one case that
            # cannot bomb.
            total += len(raw)
        else:
            # `+ 1` past the remaining budget so that a stream landing EXACTLY on the cap
            # is distinguishable from one that exceeds it, without decompressing the
            # excess. `unconsumed_tail` is non-empty precisely when the ceiling stopped
            # the decode early, which is the over-cap signal; a clean decode consumes all
            # input and leaves it empty.
            engine = zlib.decompressobj()
            try:
                decoded = engine.decompress(raw, max(limit - total, 0) + 1)
            except zlib.error as exc:
                raise PageContentUndecodable(f"page content stream could not be inflated: {exc}") from exc
            if engine.unconsumed_tail:
                raise PageContentTooLarge(f"page content stream decompresses past the {limit}-byte cap")
            if not engine.eof:
                # Input exhausted with the zlib stream still open: the stream is TRUNCATED,
                # and its valid prefix inflated cleanly. Checking `unconsumed_tail` alone
                # cannot see this -- that tail is empty precisely because every compressed
                # byte was consumed -- so without this branch a truncated stream is measured,
                # declared "sized", and handed on as if it were whole. Refused rather than
                # sized, because a length established from a prefix is not the length of the
                # content, and a page parsed from half a stream fails in the silent
                # direction: fewer operations, no error, a short page that looks complete.
                raise PageContentUndecodable(
                    "page content stream ends mid-deflate; its length cannot be established from a truncated prefix"
                )
            trailing = bytes(engine.unused_data).strip(_PDF_WHITESPACE)
            if trailing:
                # Bytes after the deflate stream's own end marker that are not PDF
                # whitespace. This function did not measure them and `ContentStream` may
                # well parse them, so the number returned would bound less than the caller
                # believes.
                #
                # The whitespace exemption is measured, not defensive: 43 of the corpus's
                # 161 content streams carry exactly one trailing b"\n" -- the EOL that PDF's
                # own stream syntax puts before `endstream` and that `/Length` need not
                # cover. Refusing on `unused_data` alone failed 43 real pages across 4 of 8
                # papers and dropped the corpus from 78,178 fragments to 34,151. A guard
                # that costs 56% of the evidence to catch a byte the format requires is
                # measuring the format rather than the document.
                raise PageContentUndecodable(
                    f"page content stream carries {len(trailing)} non-whitespace bytes past "
                    "the end of its deflate data; the size cannot be established"
                )
            total += len(decoded)
        if total > limit:
            raise PageContentTooLarge(f"page content stream decompresses past the {limit}-byte cap")
    return total


def _glyphs_drawn(show: Any) -> int:
    """How many glyphs one text-show operation drew -- i.e. how many ``Tc`` it owes.

    Neither of the two obvious answers is right, and both are wrong in the SAME
    direction, which is why this is its own function with its own test.

    * ``len(show.text)`` counts characters after the font's ``character_map`` runs, and
      that map may expand one code into several.
    * ``len(show._decoded_value)`` -- the string pypdf's own width loop iterates -- is
      not a glyph count either when the encoding is a dict. pypdf decodes those byte by
      byte through ``font.encoding[byte]``, and an entry may be a multi-character glyph
      NAME: a font whose ``/Differences`` names glyphs the standard list does not know
      turns two bytes into the eight-character string ``"/C20/C21"``.

    Measured on the eight-paper corpus: 152 shows decode to a different length than
    their operand, 10 of them with ``Tc != 0`` -- and those ten are the two largest
    errors an earlier draft of this measurement reported (231 pt and 248 pt, both of them
    this overcount rather than the defect). Counting decoded characters there
    would charge seven spacings where two are owed, so the correction would overshoot
    exactly where the original defect was worst.

    Note what this still cannot repair, because it is upstream and not about ``Tc``: on
    those same placeholder runs pypdf accumulates a WIDTH per placeholder character too,
    so their advance is unreliable whatever this returns. Re-deriving widths is not this
    module's business. Those fragments are already published as
    :attr:`GlyphMapping.UNMAPPED`, so the geometry that stays doubtful is geometry a
    caller is already told not to trust.

    **The standing report that this is a character count rather than a code count is
    true, and has no population.** Censused against a decoding that establishes the code
    count independently, over all 78,178 corpus shows: this function returns the true
    count on every one of them. 78,177 shows use a dictionary ``/Encoding``, which pypdf
    indexes BY BYTE, so one byte is one code by construction; exactly one uses a str
    encoding (``utf-16-be``), where the decoded length is the code count as well. The
    case the report describes -- a str-encoded font whose one code decodes to several
    characters -- would need a codec that expands, and none is in use here. Left as a
    known-empty risk rather than repaired, because a repair would be untested correction
    logic and the number it would change is not wrong on any document in hand.
    """
    value = show.value
    if not isinstance(value, bytes):
        return len(str(value))
    if isinstance(show.font.encoding, str):
        # A str encoding decodes the operand as a whole, so one decoded character is one
        # code -- including the multi-byte codes of a composite font.
        return len(show._decoded_value)
    return len(value)


def _pen_x_after(show: Any) -> float:
    """Absolute page x of the pen once the run is drawn, with ``Tc`` charged per glyph.

    pypdf's ``displaced_tx`` is the natural value for this and it is WRONG whenever
    character spacing is in play. It comes from ``TextStateParams.word_tx()``, which
    computes ``(font_size * total_width / 1000) + Tc + spaces * Tw`` -- one ``Tc`` for
    the whole call. The PDF text-space advance charges ``Tc`` for every glyph shown, so
    the reported right edge is short by ``(n - 1) * Tc``, horizontally scaled.

    Measured on the eight-paper corpus before this was written: 72,502 text-show
    operations, 12,529 of them with ``Tc != 0``, and **714 whose end coordinate is wrong
    by more than half a point, 222 of those containing a digit**, worst case 149.8 pt --
    a quarter of a page width, in every one of the eight papers. On one axis-label run
    the last glyph STARTS at x=435.19 and pypdf reports the run ending at x=330.00.

    The correction is applied as a delta to pypdf's own number rather than by
    re-deriving the whole advance, deliberately: font width lookup, encoding, word
    spacing and the ``Tz`` scale all stay in pypdf's hands, and the only arithmetic this
    module owns is the term pypdf undercharges. Verified against pdfplumber (pdfminer,
    sharing no code with pypdf): on the runs where both libraries return the same
    characters, the corrected end matches pdfplumber's per-character geometry exactly.

    **A third defect lives in the width lookup this deliberately borrows, and it is
    measured, recorded and NOT corrected here.** pypdf builds its width table keyed by
    ``chr(code)`` (``_collect_tt_t1_character_widths``: ``current_widths[chr(idx +
    first_char)] = int(width)``) and reads it back by the DECODED CHARACTER
    (``get_text_width``: ``character_widths.get(char, ...)``). Those are the same key only
    while the encoding is Latin-1-shaped. Where a ``/Differences`` array maps a code
    elsewhere in Unicode the lookup misses, and the miss is silent in both directions.
    Censused against ``/Widths`` read by index, which is the only reading of ``/Widths``
    the specification supports:

    * 787 of 275,031 corpus codes (0.29%) have ``decoded_char != chr(code)``;
    * 763 of them take the font's fabricated ``default`` width -- median error 0.542 pt,
      max 4.383 pt;
    * 24 HIT another code's key and silently take ITS width -- median 2.331 pt;
    * 751 shows on 44 of 75 pages, in 5 of 8 papers, mostly ``209 -> '—'`` and
      ``171 -> '´'`` in subset fonts.

    The error displaces every glyph behind it in the same show and shifts the published
    ``x_end``. It is recorded rather than repaired for one reason that is a measurement
    and not a preference: 72 of those 751 shows contain a DIGIT, and their ``x_end`` error
    is median 0.097 pt, max 0.462 pt. Owning the lookup means reading ``/Widths`` by code
    plus ``/MissingWidth``, the standard-14 metrics, ``/W`` for composite fonts and the
    Type 3 font matrix -- four correction paths, of which this corpus would exercise one,
    to move a number by under half a point where it touches evidence. Space widths take a
    different path again (``font.space_width``, fabricated when the declared entry is
    zero) and were checked separately: 6,208 space atoms, none wrong.

    Two things this does NOT claim:

    * The pen position INCLUDES the trailing ``Tc`` after the final glyph, because that
      is what the PDF operator does and what ``displaced_tx`` is documented to mean. It
      is therefore past the last glyph's ink by one character space. That stays the
      right number for THIS function -- the pen is where the next show starts -- and it
      stays fail-closed for a consumer that must not undercount a fragment's reach. It
      is no longer the only extent published: a containment test reading it refused a
      real table over 92 pt of trailing spacing the fragment never drew, so the ink
      extent is measured separately (:func:`_ink_x_end`) and travels on
      :attr:`TextFragment.ink_x_end`, with the choice of extent made per call site.
    * ``x_start`` is untouched and is separately suspect: on some mid-word shows it sits
      ~4 pt left of where pdfminer puts the first character, with every internal advance
      still exact. Different root cause, not fixed here, and not to be conflated with
      this one.

    **The standing report that this should use the full matrix is false, and measuring it
    found something else.** Scaling a text-space ``dx`` by ``a`` alone gives the x
    component and drops the y one, which sounds like a bug and is not one here. Censused
    over all 78,178 corpus shows, split on whether pypdf REWROTE the matrix in
    ``TextStateParams.__post_init__``:

    * 77,911 upright, ``b == c == 0``, where ``dx * a`` IS the complete projection;
    * 267 where pypdf multiplied by ``[1, -b, -c, 1, 0, 0]`` -- not a rotation, and it
      does not preserve length -- then recomputed ``tx`` and ``displaced_tx`` from the
      rewritten matrix. Here the delta is scaled by a factor pypdf invented and added to
      a number derived from the same invented matrix. A correct term added to a
      meaningless number is not a fix;
    * **0** with the document's own matrix and ``b != 0``, which is the only population
      where the reported defect could fire.

    What the census DID find is 2,800x larger and is not in this function. On those 267
    shows ``x_end`` is not a page x at all: on a y-axis title in
    ``10.1016-j.ijhydene.2013.10.164.pdf`` p11 this publishes ``x_end = 480.85`` where
    pdfminer -- sharing no code with pypdf -- measures the ink ending at 94.63, and the
    second label on the same page publishes 699.78 on a page 595.28 pt wide. ``x_start``
    and ``baseline_y`` are correct; only the advance is garbage. Every such show carries
    ``rotated=True``, and :mod:`carmel.services.pdf_cells` refuses to compare a rotated
    fragment's horizontal extent before any test reads it, which is why this is recorded
    rather than superseded.

    **The one part of that which is measured rather than guaranteed.**
    :attr:`TextFragment.rotated` carries pypdf's normalization flag, set only when
    ``orient()`` returns 90, 270, or a negative-``a`` 180 -- and ``orient()`` returns 0
    whenever ``m[3] > 1e-6``::

        def orient(m: list[float]) -> int:
            if m[3] > 1e-6:
                return 0
            ...

    So **every angle strictly between -90 and +90 buckets to 0**, is left un-normalized,
    and publishes ``rotated=False``. The corpus cross-tab has no such cell -- 77,911
    upright and 267 rotated, nothing between -- so the proxy holds on every document in
    hand, and `pdf_cells`' two rotated guards would simply not fire on a document that
    populated it. Such a fragment fails DIFFERENTLY, which is why it is worth stating
    separately: its ``x_end`` is correct, because the matrix is the document's own and
    ``tx + dx*a`` is the true page x. What is lost is that ``baseline_y`` records one
    scalar for a run that also climbs by ``dx*b``. Recorded rather than guarded, because
    a guard for a population no document in hand contains could only ever be tested
    against synthetic evidence.

    **Where this note lives is itself a finding.** It sat on
    :attr:`TextFragment.rotated` first, and moving it here was not editorial: a field's
    doc string is NOT a docstring. It is an ordinary ``Expr`` statement in the class
    body, so :func:`~carmel.services.semantic_deps.compute_dependency_sha`'s recursive
    docstring stripping -- which only ever removes a body's FIRST statement -- does not
    reach it, and it is hashed as code. Documenting a field therefore costs a geometry
    supersession; documenting a function costs nothing. Verified by recomputing the own
    component with and without this paragraph in each position.
    """
    glyphs = _glyphs_drawn(show)
    if glyphs < 2 or not show.Tc:
        return float(show.displaced_tx)
    undercharged = (glyphs - 1) * float(show.Tc) * (float(show.Tz) / 100.0)
    # `transform[0]` is the same factor pypdf's own `mult()` applies to a horizontal
    # displacement (`e' = dx * n[0] + n[4]`), so the delta lands in page space the way
    # the value it corrects did.
    return float(show.displaced_tx) + undercharged * float(show.transform[0])


def _glyph_substrings(show: Any) -> list[str] | None:
    """One decoded substring per GLYPH CODE the show draws, in drawing order, or ``None``.

    The per-code decomposition of the same decode :func:`_glyphs_drawn` counts and
    pypdf's ``word_tx`` charges widths over -- the three cases mirror
    ``TextStateParams.__post_init__`` exactly, because the point is to partition the
    string pypdf already produced, never to decode differently:

    * a non-``bytes`` operand: one character per code, by the same reading
      ``_glyphs_drawn`` gives it;
    * a str-encoded font: the operand decodes as a whole, so one decoded character is
      one code, composite fonts included;
    * a dict-encoded font: pypdf indexes BY BYTE, so one byte is one code, and an
      entry may be a multi-character glyph NAME -- the ``/C14``-style substrings that
      make ``len(text)`` unequal to the glyph count and make slicing ``text`` by
      glyph COUNT unsound. Returning the substrings is what makes a per-glyph
      consumer able to slice at code boundaries instead.

    ``None`` on the one path where the partition cannot be aligned to codes at all: a
    dict-encoded operand with a byte that is in no encoding entry and does not decode
    as ASCII. pypdf handles that show by re-decoding the WHOLE operand as UTF-8 with
    replacement, so its width loop runs over a string with no per-byte alignment, and
    any partition this function returned would be a guess. Callers treat ``None`` as
    "no per-glyph geometry", which degrades to the advance-based extent -- the wider,
    refusing direction.
    """
    value = show.value
    if not isinstance(value, bytes):
        return [character for character in str(value)]
    encoding = show.font.encoding
    if isinstance(encoding, str):
        return list(show._decoded_value)
    substrings: list[str] = []
    for code in value:
        if code in encoding:
            substrings.append(encoding[code])
        else:
            try:
                substrings.append(bytes((code,)).decode())
            except UnicodeDecodeError:
                return None
    return substrings


def _ink_x_end(show: Any) -> float | None:
    """Absolute page x where the last glyph's own width ends, or ``None`` if unknowable.

    :func:`_pen_x_after` deliberately includes the trailing spacing charged after the
    final glyph, because that is what the advance operator does; this subtracts exactly
    that charge and nothing else. Per ISO 32000-1 9.4.4 the charge after the last
    glyph's width is ``(Tc + Tw-if-the-glyph-is-a-space) * Th``, and both terms are
    taken the way pypdf's own ``word_tx`` accounts them -- ``Tw`` once per space
    CHARACTER of the glyph's decoded substring -- so the subtraction undoes pypdf's
    arithmetic rather than a fresh reading of the specification that could disagree
    with it.

    Verified against the real corpus fragment that motivated the field: the two-glyph
    ``Tc``-spaced show on the target table's pressure row publishes
    ``x_end = 322.469842`` and this returns ``230.451249``, reproducing the probe's
    independently measured ink extent to the last digit, with the same per-glyph
    arithmetic placing the second glyph's start at the column its neighbours occupy.

    A NEGATIVE ``Tc`` makes this larger than ``x_end`` -- with tightening spacing the
    pen ends inside the last glyph's width -- which is not clamped, because the ink
    genuinely does extend past the pen there and clamping would re-introduce the
    undercount this module exists to avoid. The OTHER direction is refused rather than
    published: spacing negative enough to walk the pen left of the show's own start
    breaks the left-to-right hull this value is read as one edge of, so a result left
    of ``x_start`` (or not a number at all) is returned as ``None``, unmeasured.

    ``None`` too for a ROTATED show, decided at the call site: the page-space
    projection here is ``transform[0]``'s, so a rotated show's value would be exactly
    as meaningless as its ``x_end`` -- and unlike ``x_end`` this field is new, so it
    can decline to exist instead of shipping a number with a warning attached.
    """
    substrings = _glyph_substrings(show)
    if not substrings:
        return None
    last = substrings[-1]
    space_char = show.font.space_char
    spaces = sum(1 for character in last if character == space_char)
    trailing = (float(show.Tc) + spaces * float(show.Tw)) * (float(show.Tz) / 100.0)
    ink = _pen_x_after(show) - trailing * float(show.transform[0])
    if not ink >= float(show.tx):  # `not >=` rather than `<` so NaN lands here too
        return None
    return ink


def _glyph_geometry(show: Any, pieces: list[str]) -> tuple[tuple[str, float, float], ...] | None:
    """Final page-space ink intervals per glyph, paired with the text pieces, or ``None``.

    The per-glyph unrolling of exactly the arithmetic :func:`_advance` and
    :func:`_pen_x_after` already apply in aggregate: each glyph's width comes from the
    same pypdf lookups ``word_tx`` uses (``get_text_width`` per decoded character,
    ``space_width`` for the space character), its ink interval is that width scaled by
    ``Tfs``/``Tz`` and projected by ``transform[0]``, and the pen then advances by the
    width PLUS the ``Tc``/``Tw`` charge -- so the recorded intervals are positions
    after ALL spacing, never raw spacing operands, and the per-glyph pen lands where
    the aggregate arithmetic says the show ends (bit-identical on the motivating
    corpus fragment; equal analytically everywhere, to float summation order).

    ``pieces`` is :func:`_text_pieces`' partition of the PUBLISHED text -- repaired
    pieces where the repair table applied -- zipped strictly against the decoded
    substrings that drive the width lookups, so the evidence names the text a consumer
    will actually slice.

    ``None``, whole-fragment rather than partial, when the partitions disagree in
    length or any interval comes out non-finite or backwards: evidence this function
    cannot vouch for is not evidence, and the consumer's fallback (the ink hull, no
    split) fails toward refusal.
    """
    substrings = _glyph_substrings(show)
    if substrings is None or len(substrings) != len(pieces):
        return None
    font = show.font
    space_char = font.space_char
    scale = float(show.font_size) / 1000.0
    tz = float(show.Tz) / 100.0
    tc = float(show.Tc)
    tw = float(show.Tw)
    a = float(show.transform[0])
    pen = float(show.tx)
    intervals: list[tuple[str, float, float]] = []
    for substring, piece in zip(substrings, pieces, strict=True):
        width = 0.0
        spaces = 0
        for character in substring:
            if character == space_char:
                width += font.space_width
                spaces += 1
            else:
                width += font.get_text_width(character)
        ink = width * scale * tz
        start = pen
        end = pen + ink * a
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            return None
        intervals.append((piece, start, end))
        pen += (ink + (tc + spaces * tw) * tz) * a
    return tuple(intervals)


class UnsupportedContentConstruct(Exception):
    """A content-stream construct whose text geometry this walker will not guess at.

    Raised rather than logged and stepped over. :func:`extract_fragments` turns any
    exception from one page into a :class:`FragmentPageFailure` plus ``lossy=True``, so
    this is the fail-closed channel: a page whose operator stream contains something
    that moves text in a way :func:`_walk_operations` does not model is reported as a
    page that could not be read, never as a page with fewer fragments. The distinction
    matters because the two are indistinguishable downstream -- a table missing its
    third column reads exactly like a two-column table.
    """


class _BudgetExhausted(Exception):
    """Internal: the per-page fragment budget ran out mid-walk.

    A separate type from :class:`UnsupportedContentConstruct` because it means the
    opposite thing. Hitting the budget is a bound working as designed and is reported as
    truncation; an unsupported construct is a refusal. Conflating them would let a
    truncated page present as a malformed one, or worse the reverse.
    """


_IDENTITY: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: The text-state operators that take a single numeric operand, mapped to the
#: :class:`_TextState` field each one sets. ``Tf`` is absent deliberately: it takes two
#: operands of different kinds and resolves a font, so it gets its own branch.
#: Text rendering modes that paint no glyphs at all: 3 is "neither fill nor stroke", 7 is
#: "add to clipping path, paint nothing". Both are how an invisible OCR layer is drawn.
_INVISIBLE_RENDER_MODES = frozenset({3.0, 7.0})

#: The eight rendering modes ISO 32000-1 table 106 defines. Anything else refuses: an
#: undefined mode is not a shade of the defined ones, and both visibility tests above
#: would read it as ordinary visible text.
_RENDER_MODES = frozenset(float(mode) for mode in range(8))

#: Rendering modes that ADD THE GLYPHS TO THE CLIPPING PATH as well as painting them
#: (ISO 32000-1 table 106: modes 4, 5 and 6; mode 7 clips without painting and is already
#: in :data:`_INVISIBLE_RENDER_MODES`).
#:
#: Refused at the show, not at the ``Tr``, so a mode that is set and then replaced before
#: any text is shown does not fail a page. The glyph shown in one of these modes is itself
#: perfectly visible -- what this walker cannot model is what it does to every LATER glyph,
#: which is now confined to the intersection of the clip with these glyph outlines. Owning
#: that means owning glyph outlines, which is further outside this module than paths are.
#:
#: Censused: `Tr` is never set to 4, 5 or 6 anywhere in the eight-paper corpus. Zero shows,
#: zero pages.
_CLIPPING_RENDER_MODES = frozenset({4.0, 5.0, 6.0})

#: The path-painting and path-ending operators, at which a clipping path marked by ``W`` or
#: ``W*`` TAKES EFFECT (ISO 32000-1 8.5.4: ``W`` only flags the current path; the graphics
#: state's clipping path is intersected with it after the path is painted or ended).
#:
#: ``n`` is in this set and is NOT ink: it ends a path without painting it, which is the
#: usual way a clip is set (``... re W n``). Every other member paints.
_PATH_PAINTING_OPERATORS: frozenset[bytes] = frozenset({b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"n"})

#: Path constructors that build something this module cannot reduce to a rectangle. ``h``
#: is deliberately absent: closing a subpath adds no geometry, so ``re h`` is still a
#: rectangle. ``re`` has its own branch.
_UNMODELLED_PATH_OPERATORS: frozenset[bytes] = frozenset({b"m", b"l", b"c", b"v", b"y"})


class _UnknownClip:
    """A clipping path is in force whose extent this module cannot model.

    A distinct sentinel rather than ``None`` because the two must never merge: ``None``
    means "no clip, publish freely" and this means "a clip exists, refuse everything".
    Collapsing them is how a fail-closed guard becomes a no-op.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "UNKNOWN_CLIP"


UNKNOWN_CLIP = _UnknownClip()

#: A clip is one of: ``None`` (none in force), an axis-aligned page-space rectangle
#: ``(x0, y0, x1, y1)``, or :data:`UNKNOWN_CLIP`.
_Clip = tuple[float, float, float, float] | _UnknownClip | None

#: How far a glyph reaches above and below its baseline, as fractions of the rendered
#: height. Nominal typographic values, NOT this font's: the true numbers live in a
#: ``/FontDescriptor`` this module does not read, and the ascent is rounded UP to a full em
#: so that erring costs a refusal rather than a publication.
#:
#: The descent is required to be inside the clip, and one earlier version of this guard did
#: not require it. That version reasoned that vertical clipping only shaves parts of glyphs
#: -- that trimming the tail off a ``y`` leaves a ``y`` -- and it was wrong in exactly this
#: project's domain: a comma hangs below the baseline, and a clip that takes it turns
#: ``1,234`` into ``1234``. Losing a separator changes a number by three orders of
#: magnitude while leaving text that looks perfectly clean, which is the same silent
#: numeric corruption the horizontal test exists to prevent. Below-baseline ink is
#: evidence.
_DESCENT_FRACTION = 0.22
_ASCENT_FRACTION = 1.0

#: How far outside the clip the nominal box above may reach and still count as contained.
#:
#: Not a fudge factor and not a threshold tuned until a page passed -- it is there because
#: the two things being compared are not the same kind of quantity. The clip rectangle is
#: exact, derived from operands in the stream. The glyph box is NOMINAL: an em-square
#: estimate built from :attr:`TextFragment.font_height` and the fractions above, never a
#: measured ink extent. Demanding exact containment of an estimate is not strictness, it is
#: false precision, and it reports a disagreement between the estimate and reality as if it
#: were a fact about the document.
#:
#: Set well below the smallest ink that can carry meaning. The comma this guard exists to
#: protect descends roughly 1.8 pt at the type sizes in the corpus, so a quarter point
#: cannot conceal one; the sub-pixel overhangs it does absorb are real and routine, because
#: producers set an axis label flush with the plot boundary and its descender crosses that
#: boundary by design. On the one corpus page where a clip is in force over text, the
#: nominal box hangs 0.05 pt below the clip -- seven times under this tolerance, and about
#: a fifth of a pixel at 300 dpi.
#:
#: What this consciously does not prove: text clipped to a horizontal band that keeps the
#: full nominal box is not caught at all, and the box is an estimate either way. This is
#: not a proof of legibility and nothing downstream may treat it as one.
_CLIP_CONTAINMENT_TOLERANCE = 0.25


def _rect_from_re(operands: list[Any], ctm: list[float]) -> tuple[float, float, float, float] | _UnknownClip:
    """One ``x y w h re`` as a page-space rectangle, or :data:`UNKNOWN_CLIP`.

    Refuses to model a ``re`` under a CTM with any rotation or skew: the operator draws a
    rectangle in USER space, and under a sheared CTM its page-space image is a
    parallelogram that no ``(x0, y0, x1, y1)`` describes. ``b`` and ``c`` are the shear
    terms of ISO 32000-1 8.3.3's matrix; a negative ``a`` or ``d`` is only a flip, which
    ``min``/``max`` below absorbs.
    """
    if len(operands) < 4:
        raise UnsupportedContentConstruct("a re with fewer than four operands")
    if abs(ctm[1]) > 1e-9 or abs(ctm[2]) > 1e-9:
        return UNKNOWN_CLIP
    x, y, width, height = (_num(value) for value in operands[:4])
    corners = [
        _mult([1.0, 0.0, 0.0, 1.0, x, y], ctm),
        _mult([1.0, 0.0, 0.0, 1.0, x + width, y + height], ctm),
    ]
    xs = [corner[4] for corner in corners]
    ys = [corner[5] for corner in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _clip_from_path(path_rects: list[tuple[float, float, float, float] | _UnknownClip], path_unknown: bool) -> _Clip:
    """What clip the current path establishes: its single rectangle, or nothing knowable.

    More than one ``re`` before a single ``W`` is ONE path with several subpaths, whose
    filled region depends on the winding rule -- a union, or a rectangle with a hole. It is
    NOT the intersection of the rectangles, and reading it as one would enlarge the region
    this module believes it may publish inside, which is the direction that publishes
    hidden text rather than the direction that refuses visible text.
    """
    if path_unknown or len(path_rects) != 1:
        return UNKNOWN_CLIP
    return path_rects[0]


def _intersect_clips(current: _Clip, incoming: _Clip) -> _Clip:
    """Intersect the clip in force with a newly established one.

    ``UNKNOWN_CLIP`` is absorbing in both directions, and that asymmetry with ordinary set
    intersection is the point: intersecting a known rectangle with an unknown region does
    NOT leave the rectangle. The result is a subset of it whose shape is unknown, and only
    a guard that refuses can say so.
    """
    if isinstance(current, _UnknownClip) or isinstance(incoming, _UnknownClip):
        return UNKNOWN_CLIP
    if current is None:
        return incoming
    if incoming is None:
        return current
    return (
        max(current[0], incoming[0]),
        max(current[1], incoming[1]),
        min(current[2], incoming[2]),
        min(current[3], incoming[3]),
    )


def _refuse_text_outside_a_clip(params: Any, clip: _Clip) -> None:
    """Refuse a show that a clip in force does not provably contain.

    PROVABLY is the whole of it. The test is on the fragment's full extent and not on its
    origin, because a clip that contains the origin can still cut away every glyph after
    the first, and this module would then publish an ``x_end`` for ink that was never laid
    down. Anything that makes the extent unknowable -- an unmodelled clip shape, rotated
    text whose box is not axis-aligned -- refuses rather than approximates.
    """
    if clip is None:
        return
    if isinstance(clip, _UnknownClip):
        raise UnsupportedContentConstruct(
            "a text-show operator under a clipping path this module cannot reduce to a rectangle"
        )
    if bool(getattr(params, "rotated", False)):
        raise UnsupportedContentConstruct("a text-show operator that is rotated under a clipping path")
    x_start = float(params.tx)
    x_end = _pen_x_after(params)
    height = abs(float(params.font_height))
    baseline = float(params.ty)
    box = (
        min(x_start, x_end),
        baseline - _DESCENT_FRACTION * height,
        max(x_start, x_end),
        baseline + _ASCENT_FRACTION * height,
    )
    slack = _CLIP_CONTAINMENT_TOLERANCE
    if box[0] < clip[0] - slack or box[1] < clip[1] - slack or box[2] > clip[2] + slack or box[3] > clip[3] + slack:
        raise UnsupportedContentConstruct("a text-show operator whose extent a clipping path does not provably contain")


_TEXT_STATE_OPS: dict[bytes, str] = {
    b"Tc": "char_spacing",
    b"Tw": "word_spacing",
    b"Tz": "horizontal_scale",
    b"TL": "leading",
    b"Ts": "rise",
}

#: Operators the walker steps over because the SPECIFICATION says they cannot move a
#: glyph -- path construction and painting, clipping, colour, the scalar graphics-state
#: parameters, shading and marked content (ISO 32000-1 Table A.1).
#:
#: This list exists so that the walker can refuse everything NOT on it. Without a final
#: ``else`` an unrecognised operator is silently stepped over, which is the same failure
#: mode as every construct guarded above: coordinates published for a stream the engine
#: did not understand. The list is drawn from the specification's operator summary rather
#: than from the corpus, deliberately -- an allowlist built from what eight papers happen
#: to contain would refuse ordinary PDFs for using an ordinary operator.
#:
#: Censused: 30 distinct unnamed operators appear in the corpus, 236,621 calls, every one
#: of them on this list. The refusal therefore costs nothing and fires only on something
#: genuinely unmodelled.
#:
#: Four families are deliberately ABSENT, so they refuse:
#:
#: * ``BI``/``ID``/``EI`` -- an inline image carries raw binary between ``ID`` and ``EI``
#:   which a naive operand parser can mistake for operators.
#: * ``BX``/``EX`` -- a compatibility section means "ignore operators you do not know",
#:   which is precisely the instruction this guard exists to disobey.
#: * ``d0``/``d1`` -- Type 3 glyph metrics, only legal inside a glyph procedure, which
#:   this walker never enters. Meeting one means the stream is not what it claims.
#: * ``sh`` is present (a shading fill paints no glyph), but ``Do`` is not: it has its own
#:   branch and its own refusal.
#:
#: ``W``/``W*`` are NOT on this list: they have their own branch, because a clipping path
#: is a visibility channel and text shown under one is refused. The number that decided
#: that is worth recording, because the obvious one is wrong by a factor of seventy:
#: clipping OPERATORS appear on 50 of the corpus's 75 pages, and for a long time this
#: comment cited that figure to argue the channel could not be refused without failing two
#: thirds of the corpus. It counts ``W`` occurrences. Clips actually IN FORCE at the moment
#: a glyph is shown are 2 shows of 26,961, on 1 page of 73 -- because the overwhelmingly
#: common shape is ``re W n`` inside ``q ... Q``, which clips a figure and is discarded
#: before any text. Refusing it costs one real page, not fifty. See
#: :func:`_walk_operations`'s ``show`` for the cost that was accepted and why.
#:
#: KNOWN GAP, and unlike clipping this one is NOT closed -- occlusion. A fill or an image
#: drawn AFTER a glyph covers it, and every operator that can do so is on this list. The
#: ordering precondition is close to universal: 22,204 painting operators follow a show on
#: 67 of 73 corpus pages. That number is deliberately not offered as reassurance OR as
#: alarm, because it measures the precondition and not the channel -- whether any of that
#: ink geometrically covers a glyph cannot be known without the path geometry this module
#: refuses to own, so it is unmeasured rather than measured-as-zero.
#:
#: There is no unblock condition to offer, and inventing an observable-sounding one would
#: be worse than saying so: nothing in this tree can distinguish a covered glyph from a
#: visible one. What this module extracts is text-show geometry from a content stream,
#: minus an enumerated set of refused invisibility constructs. It does NOT prove that a
#: human looking at the rendered page would see the glyph, and no artifact it produces may
#: be described as if it did. Closing this needs a rasterising or path-geometry oracle in
#: the test lane; until one exists the limit is a scope statement, not a defect to fix.
_IGNORED_OPERATORS: frozenset[bytes] = frozenset(
    {
        # graphics state, scalar parameters only
        b"w",
        b"J",
        b"j",
        b"M",
        b"d",
        b"ri",
        b"i",
        # path construction: only `h` is here. Closing a subpath adds no geometry, so it
        # cannot turn a rectangle into something else. `re` and the curve/line
        # constructors have their own branches -- they decide whether a clip about to be
        # established is a rectangle this module can model or an unknown region.
        b"h",
        # path painting is NOT here: see `_PATH_PAINTING_OPERATORS`, which has its own
        # branch in `_walk_operations` because a clip marked by `W` takes effect at one of
        # those operators. They remain irrelevant to text POSITION; they stopped being
        # irrelevant to text VISIBILITY.
        # colour
        b"CS",
        b"cs",
        b"SC",
        b"SCN",
        b"sc",
        b"scn",
        b"G",
        b"g",
        b"RG",
        b"rg",
        b"K",
        b"k",
        # shading
        b"sh",
        # marked content
        b"MP",
        b"DP",
        b"BMC",
        b"BDC",
        b"EMC",
    }
)


def _mult(m: list[float], n: list[float]) -> list[float]:
    """Compose two 3x2 PDF matrices: apply ``m``, then ``n``.

    Six multiply-adds of ISO 32000-1 8.3.3, owned here rather than imported from
    ``pypdf._text_extraction.mult``. Matrix composition is defined by the specification
    and not by pypdf, and the whole point of :func:`_walk_operations` is that the
    positioning arithmetic is this module's -- borrowing the multiply would put the one
    piece the engine exists to own back behind a private import.
    """
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


def _num(value: Any) -> float:
    """One numeric operand, or a refusal.

    A content stream is not required to be well formed, and an operand that is not a
    number where the specification demands one means the operator's effect is unknown.
    Defaulting it to zero would silently place every later glyph on the page.

    Two things that ARE floats but are not numbers in the sense the operator needs:

    * ``True`` is ``float(True) == 1.0``. A boolean where a coordinate belongs means the
      stream was parsed as something other than what it is, and taking it as the number 1
      hides that behind a plausible one-unit displacement.
    * ``nan`` and ``inf``. ``float("nan")`` succeeds, and a ``nan`` in the text matrix
      propagates through every later ``_mult`` to produce coordinates that compare false
      against everything -- including against the page box, so a containment test silently
      excludes the fragment rather than reporting a bad one.
    """
    if isinstance(value, bool):
        # Unreachable through pypdf's own tokenizer, which returns `true` as an OPERATOR
        # and a `BooleanObject` (not a `bool`) inside an array. Kept as a contract on this
        # function rather than as a parser guard, because `float(True)` is 1.0 and a
        # boolean silently becoming a one-unit displacement is not a failure anyone would
        # look for. Covered by a direct unit test, not by a fixture PDF.
        raise UnsupportedContentConstruct("a positioning operand that is a boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedContentConstruct("a positioning operand that is not a number") from exc
    if not math.isfinite(number):
        # Also unreachable today, and for a reason worth writing down because it is not
        # this module's to keep: pypdf refuses a numeric token longer than 64 characters
        # (`LimitReachedError`), and no shorter PDF number literal overflows to `inf`.
        # A `nan` in the text matrix propagates through every later `_mult` and compares
        # false against everything -- including the page box, so a containment test would
        # silently EXCLUDE the fragment rather than report a bad one.
        raise UnsupportedContentConstruct(f"a positioning operand that is not finite ({number})")
    return number


def _show_operand(value: Any) -> bytes:
    """One text-show operand, which must be a string of BYTES.

    ``Tj``, ``TJ``, ``'`` and ``"`` show a string. The type test here is ``bytes`` exactly,
    and everything else refuses. Two things reach these operators that are not ``bytes``,
    and each refuses for its own reason -- neither of which is the corpus:

    * a ``NameObject``. It subclasses ``str``, so the obvious ``isinstance(value, bytes |
      str)`` test admits it, and ``[(01) /Nm (23)] TJ`` then publishes a fragment whose
      text is ``/Nm`` at real coordinates -- text FABRICATED out of a token that drew
      nothing. That is the failure class this lane exists to prevent, arriving through
      a type test rather than through an inference.
    * a ``TextStringObject``, which pypdf produces when a string operand decodes as UTF-16
      or PDFDoc. It is also a ``str``, and it is worse than useless here: the decoding has
      already replaced the CODE BYTES with characters, and the font's ``/Encoding`` and
      ``/Widths`` are indexed by those bytes. Its glyphs would be measured against the
      wrong table while looking perfectly well formed.

    The two are NOT symmetric, and the difference was measured rather than assumed:

    The ``NameObject`` refusal is LIVE. A name token reaches a show operator regardless of
    how strings are decoded, so this is the guard that does real work, and it is the one
    that caught a real fabrication.

    The ``TextStringObject`` refusal is UNREACHABLE from this module's production path, and
    a corpus census is the wrong evidence for it -- so that census is deliberately not cited
    here. :func:`_page_fragments` constructs its ``ContentStream`` with ``forced_encoding=
    "bytes"``, and pypdf's ``create_string_object`` returns a ``ByteStringObject``
    UNCONDITIONALLY under that argument, for both ``(literal)`` and ``<48EX>`` strings. The
    eight-paper corpus is 100% ``ByteStringObject`` for that reason and no other: it is this
    module's own argument being honoured, not a property of those eight PDFs. Measuring it
    proves nothing, because every PDF ever written would measure the same.

    What the argument would be worth without it is the opposite of reassuring. Dropped, the
    default path decodes ``(Hello) Tj`` to a ``TextStringObject``: pypdf tries UTF-16 by BOM,
    then UTF-16 by an embedded ``NUL``, then PDFDoc, and falls back to bytes only when all
    three raise. Ordinary ASCII show strings decode, so ``TextStringObject`` is the DEFAULT,
    not the exotic case, and dropping the argument would silently re-index every glyph's
    width against a table keyed by the code bytes the decode replaced.

    So this branch is kept as a CONTRACT on the call site, in the same spirit as ``_num``'s
    boolean branch: it cannot fire today, it says so, and if the ``"bytes"`` argument is ever
    dropped it converts that into an immediate, total, loud page refusal rather than a page
    of plausible coordinates. Note for whoever meets it then: a ``TextStringObject`` carries
    ``_original_bytes``, so unwrapping it would LOOK like the fix. It is not. Restoring
    ``forced_encoding="bytes"`` is the fix; reaching into a private attribute to undo a
    decode this module asked for is how the borrowed half grows back.
    """
    if not isinstance(value, bytes):
        raise UnsupportedContentConstruct(
            f"a text-show operand of type {type(value).__name__}, which is not a string of bytes"
        )
    return value


@dataclass
class _TextState:
    """The text-state parameters of ISO 32000-1 table 105, as this walker tracks them.

    Mutable and copied wholesale on ``q``, because these are part of the GRAPHICS state:
    ``Q`` restores ``Tc``, ``Tw``, ``Tz``, ``TL``, ``Ts`` and the font along with the
    CTM. pypdf's ``TextStateManager`` saves only the font and font size across ``q``, so
    a ``Tc`` set inside a saved block leaks out of it there; here it does not.
    """

    font: Any = None
    font_size: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 100.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: float = 0.0
    fill_alpha: float = 1.0
    """``/ca``. Part of the graphics state, so ``q``/``Q`` save and restore it with the
    rest of this dataclass."""
    clip: _Clip = None
    """The clipping path in force: ``None``, an axis-aligned page-space rectangle, or
    :data:`UNKNOWN_CLIP`. Graphics state (ISO 32000-1 8.4.2), so ``Q`` restores it with the
    rest of this dataclass -- which is why ``q re W n ... Q`` around a figure costs nothing.

    A rectangle and NOT a path, which is a deliberate line rather than a simplification. A
    plot area is ``x y w h re W n``, so modelling that one case returns the figure pages
    the boolean version of this guard refused; modelling anything else means owning subpath
    accumulation, winding rules and polygon intersection, which this module does not do.
    Everything that is not a single rectangle becomes :data:`UNKNOWN_CLIP` and refuses."""
    clip_pending: bool = False
    """``W`` seen, no path-painting operator yet. Refused at a show exactly like
    :attr:`clip_active`: a stream that shows text between ``W`` and the operator that ends
    the path is malformed for the state machine above, and the safe reading of a malformed
    clip is that a clip is coming. Zero corpus population."""
    stroke_alpha: float = 1.0
    """``/CA``. Separate from :attr:`fill_alpha` because which of the two makes text
    invisible depends on the rendering mode, and conflating them refuses real pages."""


def _advance(show: Any) -> float:
    """The TEXT-SPACE displacement one show applies to the text matrix.

    ISO 32000-1 9.4.4 gives the displacement of a single glyph as::

        tx = ((w0 - Tj / 1000) * Tfs + Tc + Tw) * Th

    summed over the glyphs shown, with ``Tw`` charged only on the single-byte code 32
    and the ``Tj`` term contributed by ``TJ`` array numbers rather than by glyphs.
    pypdf's ``word_tx`` computes that sum with ``Tc`` added ONCE per call instead of
    once per glyph, which is the same undercount :func:`_pen_x_after` repairs for the
    published right edge -- and repairing it there was never enough, because the
    undercounted advance is also what the next show is positioned from.

    Deliberately expressed as pypdf's own number plus the missing term rather than as a
    fresh width sum. Font width lookup, ``/Encoding`` decoding, the space-character
    test and the ``Tz`` scale all stay in pypdf's hands: this module owns the matrix
    arithmetic and nothing else. The known consequence is that where pypdf's width
    lookup is itself wrong -- a ``/Differences`` font whose codes decode to multi-
    character glyph NAMES, where it accumulates one width per name character -- the
    advance is wrong here too. Those shows are already published as
    :attr:`GlyphMapping.UNMAPPED`.
    """
    base = float(show.word_tx(show.value))
    glyphs = _glyphs_drawn(show)
    if glyphs < 2 or not show.Tc:
        return base
    return base + (glyphs - 1) * float(show.Tc) * (float(show.Tz) / 100.0)


def _refuse_form_xobject(operands: list[Any], xobjects: Any) -> None:
    """Refuse a ``Do`` that could be drawing text, and let an image through.

    pypdf's layout-mode walker has no ``Do`` branch at all, so text inside a form
    XObject is invisible to it -- not misplaced, ABSENT. That is the more dangerous of
    the two failure modes and it is the one this module inherited.

    Recursing into the form is the complete fix and it is not built, on measurement
    rather than on taste. Censused over the eight-paper corpus: **70 ``Do`` calls on 37
    of 75 pages, and not one of them resolves to a ``/Form`` XObject** -- every one is
    an image. Recursion would therefore be a resource-dictionary walk, a ``/Matrix``
    composition, a cycle guard and a depth limit, none of which any document in hand
    would execute, tested only against fixtures written to exercise it. A refusal is
    honest at zero corpus cost, and it converts a silent hole into a recorded page
    failure. When a corpus arrives that needs the text, the refusal is what will make
    that visible.

    Everything that is not exactly an ``/Image`` refuses, not only a ``/Form``. An
    allowlist rather than a denylist because the question being asked is "can I prove
    this draws no text", and a missing, malformed or unrecognised ``/Subtype`` proves
    nothing. All 71 XObjects in the corpus are ``/Image``.
    """
    if not operands:
        raise UnsupportedContentConstruct("a /Do operator with no operand")
    name = operands[0]
    try:
        entry = xobjects.get(name) if xobjects is not None else None
        subtype = entry.get_object().get("/Subtype") if entry is not None else None
    except Exception as exc:  # noqa: BLE001 - any resolution failure is a refusal
        raise UnsupportedContentConstruct("a /Do naming an unresolvable XObject") from exc
    if entry is None:
        raise UnsupportedContentConstruct("a /Do naming an XObject the page does not declare")
    if subtype != "/Image":
        raise UnsupportedContentConstruct(
            f"a /Do on an XObject of subtype {subtype!r}, which may draw text this module does not position"
        )


@dataclass(frozen=True)
class _PageResources:
    """The three resource sub-dictionaries the walker consults, resolved once per page.

    Resolved up front rather than per operator so that an unreadable resource dictionary
    fails the page at a predictable point instead of partway through the operator stream,
    and so the walker itself stays a pure function of its operand stream plus this.
    """

    xobjects: Any = None
    """``/XObject``. ``None`` is not "no XObjects" -- it is "this page does not say", and
    a ``Do`` against it refuses either way."""

    ext_gstates: Any = None
    """``/ExtGState``. Consulted only to see whether a named state carries ``/Font``."""

    vertical_fonts: frozenset[str] = frozenset()
    """Font resource names this module refuses to position; see
    :func:`_unpositionable_fonts`."""


def _resolve_resource(page: Any, key: str) -> Any:
    resources = page.get("/Resources")
    if resources is None:
        return None
    try:
        entry = resources.get_object().get(key)
        return None if entry is None else entry.get_object()
    except Exception:  # noqa: BLE001 - an unreadable resource dict is "does not say"
        logger.debug("page /Resources %s could not be resolved", key, exc_info=True)
        return None


def _unpositionable_fonts(fonts: Any) -> frozenset[str]:
    """Font resource names whose writing mode this module will not assume is horizontal.

    The engine advances the pen in x, unconditionally. That is the scope boundary the
    user set -- no vertical writing modes -- and a boundary that is not enforced is not a
    boundary: a Type0 font with a vertical CMap advances in y, and the walker would place
    every glyph after the first at a fabricated x while raising nothing.

    Two things refuse, and the split is deliberate:

    * an ``/Encoding`` NAME ending in ``-V``, which is how the predefined vertical CMaps
      are spelled (``/Identity-V``, ``/UniJIS-UCS2-V``, ...). Reading a name is not
      reading a CMap.
    * an ``/Encoding`` that is a STREAM, i.e. an embedded CMap. Its ``WMode`` is inside
      the CMap, and reading CMaps is exactly what this module was told not to do. Most
      embedded CMaps are horizontal, so this refuses more than it must -- fail-closed on
      the side where being wrong publishes coordinates.

    Censused over the corpus: 633 ``/Type1`` font resources and one ``/Type0``, whose
    encoding is ``/Identity-H``. Nothing here refuses any document in hand.
    """
    if fonts is None:
        return frozenset()
    refused: set[str] = set()
    for name in fonts:
        try:
            encoding = fonts[name].get_object().get("/Encoding")
        except Exception:  # noqa: BLE001 - an unreadable font entry is unpositionable
            refused.add(str(name))
            continue
        if encoding is None:
            continue
        if isinstance(encoding, str):
            if encoding.endswith("-V"):
                refused.add(str(name))
        elif not hasattr(encoding, "get"):
            refused.add(str(name))
        elif encoding.get("/Type") == "/CMap" or hasattr(encoding, "get_data"):
            # An embedded CMap, dictionary or stream. `/WMode` lives inside it and this
            # module does not read CMaps.
            #
            # Written as `/Type == /CMap` and NOT as "has a /Type", which is what the
            # first cut said and which refused every Type1 font in the test suite: a
            # simple `/Encoding` dictionary carrying `/Differences` declares
            # `/Type /Encoding`, so "has a /Type" matched the overwhelmingly common
            # horizontal case. The guard was a false positive against real data while
            # passing its own reasoning -- caught by the corpus-shaped fixture in
            # `test_a_placeholder_glyph_name_is_one_glyph_not_its_spelling`, which is
            # exactly the population it would have destroyed.
            refused.add(str(name))
    return frozenset(refused)


def _refuse_optional_content(operands: list[Any]) -> None:
    """Refuse a marked-content section that makes its contents OPTIONAL.

    ``/OC ... BDC`` binds everything up to the matching ``EMC`` to an optional-content
    group, and a group can be off: the text is in the file, positioned exactly as this
    module would report it, and no reader ever sees it. That is the same hazard as
    rendering mode 3 and zero fill alpha, arriving through a third channel -- and unlike
    those two it cannot be settled from the operator alone, because whether the layer is
    visible lives in the document catalog's ``/OCProperties`` and in a viewer's own
    configuration state.

    Every other marked-content tag is ignorable and stays so. The corpus carries 606
    ``BDC`` operators across 62 pages -- ``/Figure``, ``/P``, ``/Caption``, ``/Artifact``
    and the rest of the Tagged PDF structure vocabulary -- and **not one** ``/OC``. A
    guard on ``BDC`` itself would have failed 62 of 75 pages; a guard on the tag costs
    nothing and closes the channel.
    """
    if not operands:
        raise UnsupportedContentConstruct("a BDC with no operands")
    if str(operands[0]) == "/OC":
        raise UnsupportedContentConstruct(
            "a BDC binding its contents to an optional-content group, which may be hidden"
        )


def _refuse_a_reframed_page(page: Any) -> None:
    """Refuse a page whose own dictionary moves the frame the walker measures in.

    :func:`_walk_operations` starts the CTM at the identity, which makes every coordinate
    it publishes a default user-space coordinate. Two page-level entries break that
    silently, and neither appears anywhere in the content stream the walker reads:

    * ``/Rotate``. The page is displayed turned by a multiple of 90 degrees. Coordinates
      stay in an unrotated space, so ``x_start`` is still arithmetically "correct" while
      naming a position on an axis the reader never sees. A locator built from it points
      a human at the wrong side of the page -- checkable, and wrong.
    * ``/UserUnit``. The page declares that one unit is not 1/72 inch. Every distance this
      module publishes, and every threshold compared against one, is then in the wrong
      scale by a factor nothing records.

    Both refuse rather than being applied. Applying ``/Rotate`` is a four-line matrix and
    it is still a correction, tested against nothing: all 75 corpus pages declare
    ``/Rotate 0`` and not one declares ``/UserUnit``, so a rotation branch would ship
    unexercised, while a refusal that never fires costs nothing and turns a silently
    reframed page into a recorded failure.
    """
    try:
        rotate = page.get("/Rotate")
        user_unit = page.get("/UserUnit")
    except Exception as exc:  # noqa: BLE001 - a page whose own dict is unreadable refuses
        raise UnsupportedContentConstruct("a page whose dictionary cannot be read") from exc
    if rotate is not None:
        try:
            turned = float(rotate.get_object() if hasattr(rotate, "get_object") else rotate)
        except (TypeError, ValueError) as exc:
            raise UnsupportedContentConstruct("a page whose /Rotate is not a number") from exc
        # A multiple of 360 is no rotation. Anything else, including the negative and
        # out-of-range values the specification allows, refuses.
        if turned % 360.0 != 0.0:
            raise UnsupportedContentConstruct(
                f"a page rotated by {turned:g} degrees, which this module does not reframe"
            )
    if user_unit is not None:
        try:
            unit = float(user_unit.get_object() if hasattr(user_unit, "get_object") else user_unit)
        except (TypeError, ValueError) as exc:
            raise UnsupportedContentConstruct("a page whose /UserUnit is not a number") from exc
        # On the VALUE, not on the key. `/UserUnit 1` is the default and rescales nothing,
        # so refusing its mere presence would fail a page that states explicitly what every
        # other page says by omission -- the same shape of false positive as refusing
        # `/Rotate 0` would be.
        if unit != 1.0:
            raise UnsupportedContentConstruct(
                f"a page declaring /UserUnit {unit:g}, whose distances are not in default user space"
            )


def _page_resources(page: Any) -> _PageResources:
    return _PageResources(
        xobjects=_resolve_resource(page, "/XObject"),
        ext_gstates=_resolve_resource(page, "/ExtGState"),
        vertical_fonts=_unpositionable_fonts(_resolve_resource(page, "/Font")),
    )


def _painted_invisibly(state: _TextState) -> bool:
    """Whether text shown in this state would leave no mark on the page.

    ISO 32000-1 table 106: mode 0 fills, 1 strokes, 2 does both, 4/5/6 repeat those three
    and also add to the clipping path. Modes 3 and 7 paint nothing at all and are refused
    before this is reached. Text is invisible when every operation its mode performs is
    fully transparent -- so a filled glyph needs only ``/ca``, a stroked one only ``/CA``,
    and a fill-and-stroke glyph is invisible only when BOTH are zero.

    Deliberately ``== 0`` and not a threshold. A glyph at 1% opacity is faint, not absent,
    and it is still evidence; picking a visibility cutoff would be this module inventing a
    perceptual judgement it has no basis for.
    """
    mode = state.render_mode
    fills = mode in (0.0, 2.0, 4.0, 6.0)
    strokes = mode in (1.0, 2.0, 5.0, 6.0)
    if not fills and not strokes:
        return False  # an unknown mode is not a claim of invisibility
    return (not fills or state.fill_alpha == 0.0) and (not strokes or state.stroke_alpha == 0.0)


def _apply_ext_gstate(operands: list[Any], ext_gstates: Any, state: _TextState) -> None:
    """Apply the parts of a named graphics state that bear on text, or refuse.

    An ExtGState may set the font and size without a ``Tf``, and a walker that ignores
    ``gs`` then advances using the PREVIOUS font's widths -- wrong coordinates with
    nothing raised. It may also set the alphas, and an alpha of zero paints nothing at
    all: the same hazard as rendering mode 3, arriving through the graphics state instead
    of through a text operator.

    Refusing on the operator itself is not available. The corpus carries 2,862 ``gs``
    invocations on 73 of 75 pages, so a blanket refusal would fail almost every page in
    hand. What refuses is named:

    * ``/Font`` -- zero in the corpus.
    * ``/SMask`` other than ``/None``. A soft mask can erase what is painted, and
      evaluating one means owning the mask's own content stream. All nine ``/SMask``
      entries in the corpus are ``/None``.

    The alphas are RECORDED rather than refused, and the difference is the whole reason
    this function is not a blanket alpha guard. ``/CA`` is the STROKE alpha, and 2,552 of
    the corpus's ``gs`` invocations set ``/CA 0`` on 7 pages -- while not one page in the
    corpus contains a single ``Tr`` operator, so every glyph is drawn in mode 0, filled
    only, and its stroke alpha is irrelevant to whether it is visible. A guard that read
    "any alpha of zero refuses" would have failed 7 real pages for a construct that does
    not touch their text. Which alpha matters is decided per show, by the rendering mode,
    in :func:`_walk_operations`.
    """
    if not operands:
        raise UnsupportedContentConstruct("a gs operator with no operand")
    if ext_gstates is None:
        raise UnsupportedContentConstruct("a gs naming a state the page does not declare")
    try:
        entry = ext_gstates.get(operands[0])
        resolved = None if entry is None else entry.get_object()
    except Exception as exc:  # noqa: BLE001 - any resolution failure is a refusal
        raise UnsupportedContentConstruct("a gs naming an unresolvable graphics state") from exc
    if resolved is None:
        raise UnsupportedContentConstruct("a gs naming a state the page does not declare")
    if "/Font" in resolved:
        raise UnsupportedContentConstruct("a gs that sets the font without a Tf")
    if "/SMask" in resolved:
        try:
            # RESOLVED before it is compared. `/SMask` may be an indirect reference, and
            # `str()` on one renders `IndirectObject(...)` -- which is not `/None`, so a
            # mask that is being turned OFF through a reference would refuse the page.
            mask = resolved["/SMask"].get_object()
        except Exception as exc:  # noqa: BLE001 - an unresolvable mask is a refusal
            raise UnsupportedContentConstruct("a gs whose /SMask cannot be resolved") from exc
        if str(mask) != "/None":
            raise UnsupportedContentConstruct("a gs that installs a soft mask, which may erase the text it paints")
    for key, field in (("/ca", "fill_alpha"), ("/CA", "stroke_alpha")):
        if key in resolved:
            try:
                value = resolved[key].get_object()
            except Exception as exc:  # noqa: BLE001 - an unresolvable alpha is a refusal
                raise UnsupportedContentConstruct(f"a gs whose {key} cannot be resolved") from exc
            alpha = _num(value)
            if not 0.0 <= alpha <= 1.0:
                # ISO 32000-1 table 58: a constant alpha is a number in [0, 1]. Outside it
                # the file is saying something no renderer agrees on -- clamp, ignore, or
                # error, all three occur -- and this module's own visibility test would
                # read `/ca 2` as "opaque, carry on". A state that cannot be evaluated is
                # not a state to publish coordinates from.
                raise UnsupportedContentConstruct(
                    f"a gs whose {key} is {alpha:g}, outside the [0, 1] range an alpha may take"
                )
            setattr(state, field, alpha)


def _walk_operations(
    operations: list[tuple[list[Any], bytes]],
    *,
    fonts: dict[str, Any],
    resolve_font: Any,
    params_cls: Any,
    resources: _PageResources,
    budget: int,
) -> tuple[list[Any], bool]:
    """Recompute where every text-show operation on one page actually starts.

    This is the scoped position engine. It owns exactly one thing -- the horizontal text
    positioning arithmetic of ISO 32000-1 9.4.2-9.4.4 -- and hands everything else back
    to pypdf: fonts are resolved by ``resolve_font``, operands are decoded and per-show
    quantities (``text``, ``font_height``, ``rotated``, the rotation normalisation) are
    derived by constructing a real ``TextStateParams`` around the transform computed
    here. There is no CMap reading, no ``ToUnicode`` handling and no vertical writing
    mode in this function, and there is not meant to be.

    **Why it exists.** pypdf's ``recurse_to_target_op`` is wrong about where text starts,
    in two distinct ways, both established against the SPECIFICATION rather than against
    a peer library -- eleven synthetic PDFs whose every operand and every glyph width was
    chosen so the expected origins could be computed by hand. pypdf matched on 7 of 11:

    * A show operator does not advance the pen at all. ``(01) Tj (23) Tj`` puts both runs
      at the same x, because the ``Tj`` branch appends the show and never applies a
      displacement; only a ``TJ`` array's NUMBERS displace anything. 22 sites on 11 of
      the corpus's 75 pages.
    * Within a ``TJ`` array the displacement applied between elements charges ``Tc``
      once for the whole element, so every element after the first starts short by
      ``Tc`` times the number of glyphs before it. 7,815 elements on 61 of 75 pages.

    The second defect is invisible in body text, where ``Tc`` is zero, and dominates
    exactly where this lane's evidence is: figure tick rows are drawn as ``Tc``-spaced
    runs with ``Tc`` set to the tick pitch.

    **What it does not fix.** Widths. Every width used here is the one pypdf looks up,
    so a font whose metrics pypdf gets wrong is still wrong -- see :func:`_advance`.
    This walker moves the pen correctly through the widths it is given; it does not
    check them.

    Returns the shows in stream order, and whether ``budget`` cut the walk short.
    """
    ctm: list[float] = list(_IDENTITY)
    state = _TextState()
    stack: list[tuple[list[float], _TextState]] = []
    # `None` outside a text object. A positioning or showing operator that arrives with
    # no `BT` in effect has no text matrix to act on, and inventing an identity for it
    # would place the text at the page origin rather than admit the stream is malformed.
    tm: list[float] | None = None
    tlm: list[float] | None = None
    shows: list[Any] = []
    # The CURRENT PATH. Not graphics state, so unlike `state.clip` these are plain locals
    # that `q` does not save and `Q` does not restore -- they are cleared at every
    # path-ending operator instead. Note what that means precisely, because the obvious
    # phrasing ("a path does not survive q/Q") describes a different implementation from
    # this one: a path under construction when a `q` arrives DOES survive it here, and
    # survives the matching `Q` too. Constructing a path across a save/restore boundary is
    # malformed, and the safe reading of a malformed path is the one that carries its
    # geometry forward to the clip it may still establish, rather than silently dropping
    # it and leaving `path_rects` empty -- which `_clip_from_path` would then read as
    # UNKNOWN, refusing. Both readings are safe; this one refuses less.
    # `path_unknown` covers both a non-rectangular constructor and a `re` this module
    # declined to reduce.
    path_rects: list[tuple[float, float, float, float] | _UnknownClip] = []
    path_unknown = False

    def require_text_object() -> tuple[list[float], list[float]]:
        if tm is None or tlm is None:
            raise UnsupportedContentConstruct("a text operator outside any BT/ET object")
        return tm, tlm

    def next_line(dx: float, dy: float) -> None:
        nonlocal tm, tlm
        _tm, _tlm = require_text_object()
        tlm = _mult([1.0, 0.0, 0.0, 1.0, dx, dy], _tlm)
        tm = list(tlm)

    def show(value: Any) -> None:
        nonlocal tm
        _tm, _tlm = require_text_object()
        if state.font is None:
            raise UnsupportedContentConstruct("a text-show operator before any Tf")
        if state.render_mode in _INVISIBLE_RENDER_MODES:
            # Modes 3 and 7 paint no glyphs. An OCR layer beneath a scanned page is
            # exactly this, and publishing its geometry would put an invisible copy of a
            # number in competition with the visible one at the same coordinates. Refused
            # rather than skipped: skipping drops the only text such a page has, silently.
            # Zero corpus cost -- there is not one `Tr` operator in the eight papers.
            raise UnsupportedContentConstruct(
                f"a text-show operator in rendering mode {state.render_mode:g}, which paints nothing"
            )
        if state.render_mode in _CLIPPING_RENDER_MODES:
            # Modes 4-6 paint this glyph AND add it to the clipping path, confining every
            # later glyph on the page to the intersection. Refused at zero corpus cost.
            raise UnsupportedContentConstruct(
                f"a text-show operator in rendering mode {state.render_mode:g}, "
                "which adds its glyphs to the clipping path"
            )
        if state.clip_pending:
            # `W` seen, path not yet ended. Malformed for the state machine, and the safe
            # reading of a malformed clip is that a clip is coming. Zero corpus population.
            raise UnsupportedContentConstruct("a text-show operator between a W and the operator that ends its path")
        if _painted_invisibly(state):
            # The same hazard as mode 3, reached through the graphics state instead. Which
            # alpha counts is decided HERE and not at the `gs`, because the mode in effect
            # when the state was installed need not be the mode in effect when text is
            # finally shown.
            raise UnsupportedContentConstruct(
                f"a text-show operator whose paint is fully transparent in rendering mode {state.render_mode:g}"
            )
        if len(shows) >= budget:
            raise _BudgetExhausted
        params = params_cls(
            value,
            state.font,
            state.font_size,
            state.char_spacing,
            state.word_spacing,
            state.horizontal_scale,
            state.leading,
            state.rise,
            _mult(_tm, ctm),
        )
        # After the params exist, because the test is on the fragment's published EXTENT
        # and `_pen_x_after` needs them. Before the append, because a fragment a clip does
        # not contain must never reach the list at all.
        _refuse_text_outside_a_clip(params, state.clip)
        shows.append(params)
        # The pen advances along the ORIGINAL text matrix, never along the one
        # `TextStateParams.__post_init__` may have rewritten. pypdf rewrites the
        # transform of rotated text by a non-length-preserving factor of its own
        # invention; feeding that back into the next show's position would propagate an
        # invented number down the rest of the text object.
        tm = _mult([1.0, 0.0, 0.0, 1.0, _advance(params), 0.0], _tm)

    try:
        for operands, op in operations:
            if op == b"q":
                stack.append((list(ctm), dataclasses.replace(state)))
            elif op == b"Q":
                if not stack:
                    raise UnsupportedContentConstruct("a Q with no matching q")
                ctm, state = stack.pop()
            elif op == b"cm":
                if len(operands) < 6:
                    raise UnsupportedContentConstruct("a cm with fewer than six operands")
                ctm = _mult([_num(v) for v in operands[:6]], ctm)
            elif op == b"BT":
                # A text object starts with both matrices at identity, and neither
                # survives `ET`. Nested `BT` is illegal; treating it as a reset matches
                # what the operator means where it is legal.
                tm = list(_IDENTITY)
                tlm = list(_IDENTITY)
            elif op == b"ET":
                tm = tlm = None
            elif op in (b"Td", b"TD"):
                if len(operands) < 2:
                    raise UnsupportedContentConstruct("a Td/TD with fewer than two operands")
                dx, dy = _num(operands[0]), _num(operands[1])
                if op == b"TD":
                    state.leading = -dy
                next_line(dx, dy)
            elif op == b"Tm":
                if len(operands) < 6:
                    raise UnsupportedContentConstruct("a Tm with fewer than six operands")
                require_text_object()
                tlm = [_num(v) for v in operands[:6]]
                tm = list(tlm)
            elif op == b"T*":
                next_line(0.0, -state.leading)
            elif op == b"Tf":
                if len(operands) < 2:
                    raise UnsupportedContentConstruct("a Tf with fewer than two operands")
                if str(operands[0]) in resources.vertical_fonts:
                    raise UnsupportedContentConstruct(
                        "a Tf naming a font whose writing mode this module cannot prove is horizontal"
                    )
                state.font = resolve_font(fonts, operands[0])
                state.font_size = _num(operands[1])
            elif op in _TEXT_STATE_OPS:
                if not operands:
                    raise UnsupportedContentConstruct("a text-state operator with no operand")
                setattr(state, _TEXT_STATE_OPS[op], _num(operands[0]))
            elif op == b"Tj":
                if not operands:
                    raise UnsupportedContentConstruct("a Tj with no operand")
                show(_show_operand(operands[0]))
            elif op == b"TJ":
                if not operands:
                    raise UnsupportedContentConstruct("a TJ with no operand")
                if not isinstance(operands[0], list):
                    # `TJ` takes an array. Anything else that happens to be iterable would
                    # be walked as one: a bytes operand yields INTEGERS, every one of which
                    # this loop would apply as a displacement, silently converting a string
                    # into a run of pen movements. All 9,372 corpus `TJ` operands are
                    # arrays. `ArrayObject` subclasses `list`, so this admits them.
                    raise UnsupportedContentConstruct("a TJ whose operand is not an array")
                for element in operands[0]:
                    if isinstance(element, bytes | str):
                        show(_show_operand(element))
                        continue
                    # A number in a TJ array displaces the pen by `-k/1000 * Tfs * Th`
                    # and charges NEITHER `Tc` nor `Tw` -- it is not a glyph. Applied to
                    # `tm` directly, so a run of numbers with no strings between them
                    # accumulates the way the specification says it does.
                    _tm, _tlm = require_text_object()
                    tm = _mult(
                        [
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            -_num(element) / 1000.0 * state.font_size * (state.horizontal_scale / 100.0),
                            0.0,
                        ],
                        _tm,
                    )
            elif op == b"'":
                if not operands:
                    raise UnsupportedContentConstruct("a ' with no operand")
                next_line(0.0, -state.leading)
                show(_show_operand(operands[0]))
            elif op == b'"':
                if len(operands) < 3:
                    raise UnsupportedContentConstruct('a " with fewer than three operands')
                # `aw ac string "` sets word and character spacing PERMANENTLY, not just
                # for this show, then does the implied `T*`.
                state.word_spacing = _num(operands[0])
                state.char_spacing = _num(operands[1])
                next_line(0.0, -state.leading)
                show(_show_operand(operands[2]))
            elif op == b"Tr":
                if not operands:
                    raise UnsupportedContentConstruct("a Tr with no operand")
                mode = _num(operands[0])
                if mode not in _RENDER_MODES:
                    # ISO 32000-1 table 106 defines exactly eight modes, as integers.
                    # `3.5 Tr` is not "somewhere between invisible and clip": it is a
                    # stream saying something the specification does not define, and both
                    # the invisibility test and `_painted_invisibly` would have read it as
                    # an ordinary visible mode and carried on.
                    raise UnsupportedContentConstruct(
                        f"a Tr naming rendering mode {mode:g}, which is not one of the eight defined modes"
                    )
                state.render_mode = mode
            elif op == b"gs":
                _apply_ext_gstate(operands, resources.ext_gstates, state)
            elif op == b"Do":
                _refuse_form_xobject(operands, resources.xobjects)
            elif op == b"BDC":
                _refuse_optional_content(operands)
            elif op == b"re":
                path_rects.append(_rect_from_re(operands, ctm))
            elif op in _UNMODELLED_PATH_OPERATORS:
                path_unknown = True
            elif op in (b"W", b"W*"):
                # `W` only MARKS the current path; the clip is established at the operator
                # that ends it. `W*` differs from `W` only in the winding rule, which is
                # indistinguishable for the single rectangle this module models and
                # irrelevant for everything else, since everything else is UNKNOWN anyway.
                state.clip_pending = True
            elif op in _PATH_PAINTING_OPERATORS:
                # Where a marked clip takes effect (8.5.4). Also the reason these operators
                # cannot simply stay in `_IGNORED_OPERATORS`: they are irrelevant to where
                # text goes, and load-bearing for whether it can be seen.
                if state.clip_pending:
                    state.clip = _intersect_clips(state.clip, _clip_from_path(path_rects, path_unknown))
                    state.clip_pending = False
                # Cleared on EVERY path-ending operator, marked or not. Otherwise a `re`
                # that was painted and finished stays in the list and attaches itself to a
                # later, unrelated `W`, which would hand that clip a rectangle drawn for
                # something else.
                path_rects = []
                path_unknown = False
            elif op not in _IGNORED_OPERATORS:
                # The final `else` this walk did without. Everything above is either
                # modelled or refused by name; without this an operator that is NEITHER --
                # an inline image, a compatibility section, a Type 3 glyph-metric operator,
                # a corrupted token -- was stepped over in silence, and every fragment after
                # it was published from a state the engine could not vouch for.
                raise UnsupportedContentConstruct(
                    f"an operator this walker does not model: {op.decode('latin-1', 'replace')!r}"
                )
    except _BudgetExhausted:
        return shows, True
    return shows, False


def _page_fragments(
    page: Any,
    page_number: int,
    engine: tuple[Any, ...],
    budget: int,
    verdicts: dict[str, dict[str, GlyphVerdict]] | None = None,
    glyph_budget: int = MAX_PDF_GLYPH_INTERVALS,
    font_budget: int = MAX_FONT_PROGRAM_BYTES_PER_DOCUMENT,
) -> tuple[list[TextFragment], bool, int]:
    """Recover every text-show operation on one page, with absolute geometry.

    ``verdicts`` is the glyph verdict registry ALREADY narrowed to this document by
    :func:`extract_fragments` -- font-program sha256 to {glyph name: verdict}. It is ``None``
    or empty only for a document embedding NONE of the registry's programs; because a
    FONT_PROGRAM-scoped verdict matches every document embedding its program, any registry
    holding one such verdict makes ``verdicts`` non-empty for every document, and the per-show
    font hashing below then runs for all of them (bounded by ``font_budget``).

    ``glyph_budget`` bounds the per-glyph interval entries THIS CALL may record; like
    ``budget`` it is the document-wide cap minus what earlier pages already used, so
    a single page cannot spend what the document has left.

    ``font_budget`` is the same shape for font-program inflate: the document-wide
    :data:`MAX_FONT_PROGRAM_BYTES_PER_DOCUMENT` minus what earlier pages already inflated. The
    third return value is how many decompressed font-program bytes THIS page charged, so the
    caller can carry the running total forward and a many-font document cannot inflate an
    unbounded total one bounded stream at a time.

    Stops after ``budget`` fragments and reports that it did, so a single page cannot
    exhaust :data:`MAX_PDF_FRAGMENTS`-worth of memory on its own.

    The budget bounds BOTH accumulating lists, not only the fragments converted at the
    end. Bounding the conversion alone does not work: one show becomes at most one
    fragment, so a page emitting ten million shows costs ten million retained
    ``TextStateParams`` before the conversion loop runs at all, and the cap then fires
    after the damage.

    :func:`_walk_operations` checks the budget at the point where a show is CONSTRUCTED,
    including inside a ``TJ`` array, so unlike the group-at-a-time walker this replaced
    there is no longer any nesting level at which shows accumulate unbounded. What the
    fragment budget still does not bound, stated exactly rather than glossed, because a
    comment that overstates a guard is worse than no guard: ``ContentStream``
    materialises the entire operation list up front, inside pypdf, before this function
    sees anything.

    That scales with the page's decompressed content-stream size and nothing else, which
    is why that size -- and not either of them individually -- is what gets capped
    below. See :data:`MAX_PAGE_CONTENT_BYTES` for the measurement it is set from, and
    for the one thing it still does not bound.
    """
    resolve_font, params_cls, content_stream = engine

    _refuse_a_reframed_page(page)
    contents = page.get("/Contents")
    if contents is None:
        return [], False, 0
    resolved = contents.get_object()
    # The cap is enforced INSIDE, and no second check follows it here. The measurement
    # and the refusal used to be two steps -- measure, then compare -- and that shape is
    # what let the decode allocate the whole stream before the comparison could run. A
    # belt-and-braces `> MAX` here would now be unreachable, and an unreachable guard is a
    # silent no-op that reads like protection.
    #
    # These two lines inflate the same bytes TWICE, and the duplication is the guard.
    # Measured by attributing decompressed bytes over the corpus: the two paths are equal
    # to the byte and to the call on all 8 documents (e.g. 1.104 MB in 18 calls each).
    # The obvious optimisation -- have `_decoded_content_length` return the data it just
    # decompressed, or measure via `get_data()` and reuse pypdf's cache -- is the exact
    # shape of the bomb this guard closed: either one puts an unbounded inflation FIRST.
    # pypdf's own inflation on the next line is safe only because it is second, and it is
    # only second because nothing here keeps its output.
    _decoded_content_length(resolved, MAX_PAGE_CONTENT_BYTES)
    content = content_stream(resolved, page.pdf, "bytes")

    shows, stopped_early = _walk_operations(
        content.operations,
        fonts=page._layout_mode_fonts(),
        resolve_font=resolve_font,
        params_cls=params_cls,
        resources=_page_resources(page),
        budget=budget,
    )

    fragments: list[TextFragment] = []
    glyphs_recorded = 0
    font_bytes_used = 0
    # Font-program digests, hashed once per font OBJECT rather than once per show:
    # `_layout_mode_fonts` builds one `Font` per resource name per page, so identity
    # is a safe cache key for the duration of this call and no longer.
    font_sha_cache: dict[int, str | None] = {}
    for show in shows:
        if len(fragments) >= budget:
            return fragments, True, font_bytes_used
        if any(not hasattr(show, attr) for attr in _REQUIRED_PARAM_ATTRS):
            # Belt-and-braces against a pypdf change that slipped past `_engine`.
            # `_EngineMismatch` rather than a plain error: this must abort the WHOLE
            # extraction as unavailable, not degrade one page to lossy.
            raise _EngineMismatch("pypdf TextStateParams is missing a required attribute")
        text = show.text
        if not text:
            continue
        rotated = bool(show.rotated)
        pieces = _text_pieces(show)
        mapping = GlyphMapping.UNMAPPED if _UNMAPPED_MARKER_RE.search(text) else GlyphMapping.MAPPED
        if verdicts and pieces is not None:
            # Attempted whether the show is UNMAPPED or MAPPED: the fault the registry also
            # covers -- a symbol font decoding a phi to a literal 'f' -- surfaces MAPPED, so
            # gating on UNMAPPED would skip exactly the mis-decoded glyphs.
            #
            # `verdicts` is non-empty whenever ANY FONT_PROGRAM-scoped verdict is registered,
            # because such a verdict matches every document embedding its program -- so this
            # per-show hashing runs on the fonts of unregistered PDFs too, not only the
            # registered ones. `_font_program_sha256` therefore bounds its own inflate, and
            # `font_budget` bounds the CUMULATIVE inflate across every distinct font on the
            # page: the smaller of the per-stream ceiling and the document's remaining budget
            # is passed as the limit, and once the budget is spent every further font hashes to
            # `None` and its verdicts simply do not apply. The DOCUMENT-scope-first ordering
            # that used to guarantee "these bytes are pinned in full" no longer does.
            if id(show.font) not in font_sha_cache:
                remaining = font_budget - font_bytes_used
                font_sha, inflated = _font_program_sha256(show.font, min(MAX_FONT_PROGRAM_BYTES, remaining))
                font_sha_cache[id(show.font)] = font_sha
                font_bytes_used += inflated
            font_sha = font_sha_cache[id(show.font)]
            for_this_font = verdicts.get(font_sha) if font_sha is not None else None
            if for_this_font:
                judged = _judged_pieces(pieces, for_this_font)
                if judged is not None:
                    pieces, mapping = judged
                    text = "".join(pieces)
        glyphs = None if rotated or pieces is None else _glyph_geometry(show, pieces)
        if glyphs is not None:
            if glyphs_recorded + len(glyphs) > glyph_budget:
                # Truncate BEFORE this fragment rather than shipping it stripped of
                # its evidence: a fragment without its partition would silently lose
                # exactly the sub-fragment structure the field exists to carry. Same
                # channel as the fragment cap; `build_inventory` refuses truncated
                # documents wholesale.
                return fragments, True, font_bytes_used
            glyphs_recorded += len(glyphs)
        fragments.append(
            TextFragment(
                page=page_number,
                text=text,
                x_start=float(show.tx),
                x_end=_pen_x_after(show),
                baseline_y=float(show.ty),
                font_height=float(show.font_height),
                rotated=rotated,
                glyph_mapping=mapping,
                ink_x_end=None if rotated else _ink_x_end(show),
                glyph_intervals=glyphs,
            )
        )
    return fragments, stopped_early, font_bytes_used


def extract_fragments(data: bytes) -> FragmentExtraction:
    """Extract every text-show fragment from ``data``, with absolute page geometry.

    Page numbers come from ``carmel.agents.tools.extract._classify_pdf_page`` rather
    than from ``enumerate(reader.pages)``. That is not a stylistic preference. pypdf's
    ``reader.pages`` walks ``/Kids`` without checking ``/Type``, so a linearized PDF
    can have its LINEARIZATION PARAMETER DICTIONARY counted as a page -- observed on
    real corpus papers -- which shifts every later page index by one. A locator citing
    such an index sends a human to the wrong page while looking perfectly checkable,
    and it is the numbering disagreement, not the crash, that does the damage. The
    shipped text lane already solved this; a second filter that disagreed with it
    would be worse than none, so this reuses that one classifier.

    Never raises for a malformed document: returns a degraded
    :class:`FragmentExtraction` instead, matching how ``_extract_pdf`` degrades.
    """
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        # ONLY a module that is genuinely not there is ENGINE_ABSENT, and the name has
        # to be checked: an installed pypdf missing a transitive dependency raises this
        # same class naming THAT dependency, and calling it "the optional extra is not
        # installed" would repeat the original defect one layer down. Measured, not
        # assumed -- absent module and a poisoned `sys.modules` entry both raise
        # ModuleNotFoundError(name="pypdf"), a missing transitive dep raises it naming
        # the dep, and an import-time crash or a missing `PdfReader` does not raise
        # this class at all.
        if exc.name != "pypdf":
            logger.warning("pypdf is installed but its import failed: %s", exc)
            return FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_REFUSED)
        return FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_ABSENT)
    except Exception:
        # Present and broken: a crash at import time, or a package that no longer
        # exports `PdfReader`. An alarm, not a supported configuration.
        logger.warning("pypdf is installed and could not be imported; fragments unavailable", exc_info=True)
        return FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_REFUSED)

    from carmel.agents.tools.extract import (
        MAX_PDF_PAGES,
        _classify_pdf_page,
        _describe_page_error,
        _PageKind,
        _quiet_pypdf,
    )

    engine = _engine()
    if engine is None:
        return FragmentExtraction(lossy=True, status=FragmentAvailability.ENGINE_REFUSED)

    # NOT a second `importlib.metadata.version("pypdf")` call, which is what this was.
    #
    # `_engine()` has already returned non-None above, and it only does that after reading
    # that exact metadata entry and refusing unless it equals _PINNED_PYPDF_VERSION. So a
    # re-read here could only ever produce the same string -- while adding two ways to be
    # wrong that the constant does not have. It sat OUTSIDE the `try` below, so a metadata
    # failure raised `PackageNotFoundError` straight out of a function whose docstring
    # promises it never raises for a malformed document; and being a second read of a
    # mutable source, it could in principle disagree with the value the gate approved,
    # recording on the artifact a version that was never the one admitted.
    #
    # Recording the gate's own constant says exactly what is true and no more: this
    # extraction ran against the pinned pypdf, because nothing else gets this far.
    version = _PINNED_PYPDF_VERSION
    # The glyph verdict registry, narrowed to THIS document before any page is read:
    # font-program sha256 -> {glyph name: verdict}. A verdict applies when it is
    # FONT_PROGRAM-scoped (any document) or DOCUMENT-scoped to this exact document sha256,
    # the same whole-input identity every stored artifact uses. A collision inside one
    # font+glyph is impossible by construction -- `_assert_no_overlapping_verdicts` rejects
    # the registry at import if two entries could both match -- so the setdefault below is
    # never a silent last-write-wins. `verdicts` is non-empty for EVERY document as soon as one
    # FONT_PROGRAM-scoped verdict is registered (it matches any document embedding its program),
    # so `_page_fragments` then hashes this document's fonts to find out whether any of its
    # programs are the registered one; that per-font work is what `font_budget` bounds.
    verdicts: dict[str, dict[str, GlyphVerdict]] = {}
    if _GLYPH_VERDICTS:
        document_sha256 = hashlib.sha256(data).hexdigest()
        for verdict in _GLYPH_VERDICTS:
            if verdict.applies_to(document_sha256):
                verdicts.setdefault(verdict.font_program_sha256, {})[verdict.glyph_name] = verdict
    fragments: list[TextFragment] = []
    glyphs_recorded = 0
    font_bytes_used = 0
    failures: list[FragmentPageFailure] = []
    lossy = False
    truncated = False
    try:
        with _quiet_pypdf():
            reader = PdfReader(io.BytesIO(data))
            page_number = 0
            for page in reader.pages:
                # Classify BEFORE touching any page attribute: `page.mediabox` raises
                # TypeError on a phantom entry, which has no /MediaBox to resolve.
                kind = _classify_pdf_page(page)
                if kind is _PageKind.PHANTOM:
                    continue
                page_number += 1
                if page_number > MAX_PDF_PAGES:
                    # Same cap, counted the same way, as the text lane. Sharing it is
                    # the point: if one lane stopped at 2000 real pages and the other
                    # walked on, a fragment could carry a page number the text lane
                    # says does not exist, and the two provenance stories would
                    # disagree about the same document.
                    truncated = True
                    lossy = True
                    break
                if len(fragments) >= MAX_PDF_FRAGMENTS or glyphs_recorded >= MAX_PDF_GLYPH_INTERVALS:
                    # BEFORE parsing, not after. A page that ended exactly ON the cap
                    # leaves a zero budget, and entering with it would pay for the whole
                    # content stream only to report truncation on the way out.
                    truncated = True
                    lossy = True
                    break
                try:
                    page_fragments, hit_budget, page_font_bytes = _page_fragments(
                        page,
                        page_number,
                        engine,
                        MAX_PDF_FRAGMENTS - len(fragments),
                        verdicts,
                        MAX_PDF_GLYPH_INTERVALS - glyphs_recorded,
                        MAX_FONT_PROGRAM_BYTES_PER_DOCUMENT - font_bytes_used,
                    )
                except _EngineMismatch:
                    # Not a page failure. The engine is wrong, so nothing extracted
                    # from this document can be relied on. Lands in the dedicated
                    # clause below as ENGINE_CONTRADICTED_GATE, never as a document
                    # verdict: reaching here means `_engine` approved an engine that
                    # then broke the same contract, so the defect is in the gate.
                    raise
                except Exception as exc:
                    logger.debug("fragment extraction failed on page %d", page_number, exc_info=True)
                    failures.append(FragmentPageFailure(page=page_number, error=_describe_page_error(exc)))
                    lossy = True
                    continue
                fragments.extend(page_fragments)
                glyphs_recorded += sum(len(f.glyph_intervals) for f in page_fragments if f.glyph_intervals)
                font_bytes_used += page_font_bytes
                if kind is _PageKind.UNINSPECTABLE:
                    # RECORDED, not merely counted as `lossy`, and worded exactly as the
                    # text lane words it. A consumer asking "is page N sound?" reads
                    # `page_failures`, because `lossy` is a whole-document flag that
                    # cannot say WHICH page; a structural uncertainty visible only as
                    # `lossy` would let a per-page gate pass this page while the text
                    # lane records it as uncertain. Two lanes disagreeing about the same
                    # page is the failure this module exists to avoid.
                    #
                    # Success path only, mirroring the text lane: the `except` above
                    # already recorded this page, so this cannot double-record it.
                    failures.append(FragmentPageFailure(page=page_number, error=_UNINSPECTABLE_PAGE_ERROR))
                    lossy = True
                if hit_budget:
                    truncated = True
                    lossy = True
                    break
    except _EngineMismatch:
        # ORDER IS CONTRACT. `_EngineMismatch` is an `Exception` subclass, so swapping
        # these two clauses does not fail to compile and does not fail a test that only
        # checks `available` -- it silently refiles "the front-door gate is incomplete"
        # as "this document is a bit odd", which is the one misattribution this whole
        # taxonomy exists to prevent. Pinned by
        # `test_the_engine_clause_must_be_caught_before_the_general_one`.
        logger.warning("pypdf contradicted the capability gate mid-walk; fragments unavailable")
        return FragmentExtraction(
            lossy=True, status=FragmentAvailability.ENGINE_CONTRADICTED_GATE, pypdf_version=version
        )
    except Exception:
        return FragmentExtraction(lossy=True, status=FragmentAvailability.READER_WALK_FAILED, pypdf_version=version)

    return FragmentExtraction(
        fragments=tuple(fragments),
        lossy=lossy,
        status=FragmentAvailability.AVAILABLE,
        page_failures=tuple(failures),
        truncated=truncated,
        pypdf_version=version,
    )
