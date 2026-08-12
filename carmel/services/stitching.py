"""Deterministic REFUTATION of a scalar claim stitched from unrelated spans.

A :class:`~carmel.schemas.datasets.GroundedScalarClaim` grounds its label, its
value and its unit as three INDEPENDENT quotes, for a good reason: a single ref
can "verify" a number while the label silently came from somewhere else
entirely, so each half carries its own provenance. The cost of that split is
that nothing checks the three belong to each other. From the sentence

    "The initial temperature was 823 K and the pressure was held at 1.2 atm"

a caller can submit label ``"pressure"``, value ``"823"``, unit ``"atm"``: every
quote IS an exact located substring of the authenticated document, replay
reports VERIFIED with zero findings, and 823 atm is an ordinary shock-tube
pressure, so no range or plausibility check catches it either.

**What this module does and does not claim.** It REFUTES; it never verifies.
Grounding proves LOCATION, never MEANING, and no amount of arithmetic over
character offsets can prove that a paper predicates a label of a number. A
``None`` return means only "a specific, named refutation was attempted and did
not fire" -- it is not evidence that the claim is true, and it must never be
reported as ``VERIFIED``.

**The rule.** The window is DERIVED as the span covering the three locators the
caller already committed to -- deliberately NOT a new caller-supplied field,
because a caller-chosen window is policy injection: the caller could search the
whole document for whichever window happens to make their claim pass. Inside
that window there must be EXACTLY ONE unit-bearing numeral; it must be the
claimed value and unit, compared by OFFSET rather than by string; and its unit
spelling must denote the declared quantity under the same versioned table the
rest of the pipeline admits units against.

**Why uniqueness over ANY quantity kind, not just the declared one.** The
narrower "exactly one pair OF THE DECLARED KIND" rule refutes the stitch above,
but it blesses a second fabrication of the same family: label ``"temperature"``
over a value in atm, which has exactly one PRESSURE pair in its window and sails
through. Closing that by classifying the LABEL would need a label lexicon, and
real labels are ``"P1"``, ``"bore"``, ``"initial pressure"`` -- unlexiconable,
with a fail-closed unknown-label rule zeroing the yield. Any-kind uniqueness
closes both holes with no label lexicon at all, because a window spanning a
label and a foreign quantity's value necessarily also spans the numeral that
label actually belongs to.

Measured against the real 8-paper corpus before this module was written: the
derived window admits 95.5% of temperature and 92.7% of pressure claims, where
the same test scoped to a whole sentence admits 54.5%. The tighter window is
what buys the yield back; sentence scope was measured and rejected.

**Known residues, stated rather than implied.** A one-sided bound ("above
60 cm/s") has exactly one unit-bearing numeral in its window and PASSES -- it is
a method-capability threshold, not a condition, and separating the two is prose
semantics this module deliberately does not attempt. Shared spellings (``%``,
``1``) cannot discriminate mole fraction from equivalence ratio from a relative
uncertainty. Both are refusals this gate does not make; neither is a reason to
weaken the refusals it does.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from carmel.schemas.datasets import CharSpanLocator, GroundedScalarClaim
from carmel.services.numeric import NUMERAL_CANDIDATE_RE
from carmel.services.units import TABLE_V1, ConversionTable, QuantityKind

__all__ = [
    "StitchGateUnrunnable",
    "StitchRefutation",
    "UnitBearingNumeral",
    "iter_unit_bearing_numerals",
    "refute_stitched_claim",
    "refute_stitched_scalar",
    "unit_spellings",
]


_LINE_SEPARATORS = frozenset("\n\r\v\f  ")
"""Whitespace that ends a LINE rather than separating tokens within one.

Everything else that :meth:`str.isspace` accepts binds a unit to its numeral --
notably U+00A0 NO-BREAK SPACE and U+2009 THIN SPACE, both of which real PDF
extraction emits between a number and its unit.
"""


@dataclass(frozen=True, slots=True)
class UnitBearingNumeral:
    """One ``number + unit`` construct located in a window of extracted text."""

    value_start: int
    value_end: int
    unit_start: int
    unit_end: int
    value_text: str
    unit_text: str
    quantities: frozenset[QuantityKind]
    """Every quantity this unit SPELLING can denote in the table it was read
    against. A set, not a single kind, because spellings genuinely collide --
    ``"%"`` is a mole fraction and a dilution and a relative uncertainty -- and
    collapsing that to one kind would be the module inventing a fact."""


@dataclass(frozen=True, slots=True)
class StitchGateUnrunnable:
    """The gate could not run at all -- never conflated with a refutation.

    A claim whose three grounds are not all character spans into one node's
    extracted text gives this gate no window to read, so it has NOT been
    checked. That is UNVERIFIABLE, and a caller must not report it as a clean
    pass: "no refutation fired" and "no refutation was attempted" are the two
    states this codebase most insists on keeping apart.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class StitchRefutation:
    """A named, specific reason a scalar claim's three spans do NOT cohere.

    Carries the window it read and what it found there, so the refusal can be
    reported without the reader having to re-derive it. This is always a
    definite disagreement -- a check that RAN and concluded -- so a caller
    turning it into a replay finding must categorize it FAILED, never
    UNVERIFIABLE. The two are never conflated in this codebase, and a refuted
    fabrication filed as "could not check" would be the more dangerous of the
    two mistakes.
    """

    reason: str
    window_start: int
    window_end: int
    found: tuple[str, ...]


def unit_spellings(table: ConversionTable) -> Mapping[str, frozenset[QuantityKind]]:
    """Every raw unit spelling ``table`` admits, mapped to the quantities it can denote.

    Built from the table's own rules and aliases rather than from a list
    maintained here, so a unit this module can see is exactly a unit the
    pipeline can admit. A second, hand-kept list would drift from the admission
    gates, and a spelling present in one and absent from the other is precisely
    the inconsistency that lets a claim pass one check and fail another.
    """
    spellings: dict[str, set[QuantityKind]] = {}
    for rule in table.rules:
        unit = rule.unit if rule.kind == "identity" else rule.from_unit
        spellings.setdefault(unit, set()).add(rule.quantity)
        if rule.kind != "identity":
            spellings.setdefault(rule.to_unit, set()).add(rule.quantity)
    for alias in table.aliases:
        spellings.setdefault(alias.raw, set()).add(alias.quantity)
        spellings.setdefault(alias.normalized, set()).add(alias.quantity)
    return {unit: frozenset(kinds) for unit, kinds in spellings.items()}


def _is_composite_endpoint(text: str, start: int) -> bool:
    """Is the numeral at ``start`` one endpoint of a slash composite?

    :data:`~carmel.services.numeric.NUMERAL_CANDIDATE_RE` folds a hyphen/en-dash
    range into ONE match, so ``"1-5 atm"`` is already a single (unparseable)
    numeral. A SLASH composite is not folded: ``"1.0/1.5 atm"`` yields two
    matches, the first of which has no unit after it, leaving the second looking
    like the window's sole construct. Reported by Codex round 96.
    """
    cursor = start - 1
    while cursor >= 0 and text[cursor].isspace() and text[cursor] not in _LINE_SEPARATORS:
        cursor -= 1
    if cursor < 0 or text[cursor] != "/":
        return False
    cursor -= 1
    while cursor >= 0 and text[cursor].isspace() and text[cursor] not in _LINE_SEPARATORS:
        cursor -= 1
    return cursor >= 0 and text[cursor].isdigit()


def iter_unit_bearing_numerals(
    text: str,
    *,
    window_start: int,
    window_end: int,
    table: ConversionTable = TABLE_V1,
) -> Iterator[UnitBearingNumeral]:
    """Yield every ``number + unit`` construct fully inside ``[window_start, window_end)``.

    Numerals come from :data:`~carmel.services.numeric.NUMERAL_CANDIDATE_RE` --
    the same grammar the grounding and normalization paths use -- rather than a
    regex written here, so what this gate counts as a number is what the rest of
    the pipeline counts as a number.

    A unit must follow its numeral separated by nothing but INTRA-LINE
    whitespace. That is any whitespace character except the line separators in
    :data:`_LINE_SEPARATORS`, rather than only space and tab: PDF text
    extraction routinely emits U+00A0 NO-BREAK SPACE and U+2009 THIN SPACE
    between a number and its unit, and treating those as "no construct here"
    would silently refuse honest claims -- a fail-closed direction, but a wrong
    one, because the number and unit really are bound in the source.

    A line break does NOT bind, deliberately: extracted text wraps mid-sentence,
    and a numeral ending one line has no reliable relationship to a token
    beginning the next. Longest spellings match first, so ``"cm/s"`` is never
    read as the ``"cm"`` its prefix spells.
    """
    spellings = unit_spellings(table)
    by_length = sorted(spellings, key=len, reverse=True)
    for match in NUMERAL_CANDIDATE_RE.finditer(text, window_start, window_end):
        if match.end() > window_end:
            continue
        if _is_composite_endpoint(text, match.start()):
            # "1.0/1.5 atm" states a PAIR. Counting its second endpoint as the
            # window's sole construct would let a composite value be stored as
            # a scalar -- the COMPOSITE_VALUE refusal squeezed into a number.
            # Yielding nothing makes the window construct-less, which the
            # caller refuses: the safe direction.
            continue
        cursor = match.end()
        while cursor < window_end and text[cursor].isspace() and text[cursor] not in _LINE_SEPARATORS:
            cursor += 1
        for unit in by_length:
            stop = cursor + len(unit)
            if stop > window_end or text[cursor:stop] != unit:
                continue
            # Refuse a spelling that is merely the prefix of a longer word:
            # "5 minutes" must not read as 5 min, and "300 Kelvin-corrected"
            # must not read as 300 K.
            if stop < len(text) and (text[stop].isalnum() or text[stop] == "_"):
                continue
            yield UnitBearingNumeral(
                value_start=match.start(),
                value_end=match.end(),
                unit_start=cursor,
                unit_end=stop,
                value_text=match.group(),
                unit_text=unit,
                quantities=spellings[unit],
            )
            break


def refute_stitched_claim(
    claim: GroundedScalarClaim,
    text: str,
    *,
    table: ConversionTable = TABLE_V1,
) -> StitchRefutation | StitchGateUnrunnable | None:
    """Run :func:`refute_stitched_scalar` over a whole claim's three grounds.

    The single entry point the producer and the replayer both call, so the rule
    cannot drift between the lane that writes claims and the lane that checks
    them. A producer-only gate would be bypassable by any stored or forged
    envelope, which is the residue P0-c left and this deliberately does not
    repeat.
    """
    refs = (claim.label_ref, claim.value.value_ref, claim.value.unit_ref)
    names = ("label_ref", "value.value_ref", "value.unit_ref")

    spans: list[CharSpanLocator] = []
    for name, ref in zip(names, refs, strict=True):
        locator = ref.locator
        if not isinstance(locator, CharSpanLocator):
            return StitchGateUnrunnable(
                reason=(
                    f"{name} is located by {locator.kind.value}, not a character span, so "
                    "there is no window of extracted text to read. The stitching gate did not "
                    "run for this claim and its label/value association is UNCHECKED"
                )
            )
        spans.append(locator)
    label_locator, value_locator, unit_locator = spans

    nodes = {ref.node_id for ref in refs}
    if len(nodes) != 1:
        return StitchGateUnrunnable(
            reason=(
                f"this claim's three grounds span {len(nodes)} different source nodes "
                f"({', '.join(sorted(nodes))}). Character offsets index into ONE node's "
                "extracted text, so no window covers all three and the gate cannot run"
            )
        )

    spaces = {locator.text_space for locator in spans}
    if len(spaces) != 1:
        return StitchGateUnrunnable(
            reason=(
                f"this claim's three grounds index into {len(spaces)} different text spaces "
                f"({', '.join(sorted(space.value for space in spaces))}). Offsets from "
                "different spaces are not comparable, so no window covers all three"
            )
        )

    return refute_stitched_scalar(
        text,
        label_span=(label_locator.start, label_locator.end),
        value_span=(value_locator.start, value_locator.end),
        unit_span=(unit_locator.start, unit_locator.end),
        quantity_kind=claim.value.quantity_kind,
        table=table,
    )


def refute_stitched_scalar(
    text: str,
    *,
    label_span: tuple[int, int],
    value_span: tuple[int, int],
    unit_span: tuple[int, int],
    quantity_kind: QuantityKind,
    table: ConversionTable = TABLE_V1,
) -> StitchRefutation | None:
    """Attempt to refute that ``label_span`` predicates ``value_span``/``unit_span``.

    Returns a :class:`StitchRefutation` when the three spans provably do not
    cohere, and ``None`` when this specific refutation did not fire. ``None`` is
    NOT a verification -- see the module docstring.

    ``quantity_kind`` is checked against the located unit's own table entry, so
    a claim declaring PRESSURE over a span whose only unit is ``"K"`` is refuted
    without any inspection of what the label says.
    """
    window_start = min(label_span[0], value_span[0], unit_span[0])
    window_end = max(label_span[1], value_span[1], unit_span[1])
    found = tuple(iter_unit_bearing_numerals(text, window_start=window_start, window_end=window_end, table=table))
    rendered = tuple(f"{n.value_text} {n.unit_text}" for n in found)

    if len(found) != 1:
        detail = (
            "none at all -- the claimed unit may be one this table does not model, which is a refusal and never a pass"
            if not found
            else f"{len(found)}: {', '.join(rendered)}"
        )
        return StitchRefutation(
            reason=(
                f"the span covering this claim's label, value and unit "
                f"[{window_start}, {window_end}) contains {detail}. A scalar claim is "
                "admissible only when the window covering its three grounds holds exactly "
                "one number+unit construct, because with two or more nothing in the text "
                "says which one the label predicates -- and with none there is no located "
                "measurement to predicate at all"
            ),
            window_start=window_start,
            window_end=window_end,
            found=rendered,
        )

    only = found[0]
    if (only.value_start, only.value_end) != value_span or (
        only.unit_start,
        only.unit_end,
    ) != unit_span:
        return StitchRefutation(
            reason=(
                f"the only number+unit construct in [{window_start}, {window_end}) is "
                f"{rendered[0]!r} at value [{only.value_start}, {only.value_end}) unit "
                f"[{only.unit_start}, {only.unit_end}), but this claim grounds its value at "
                f"{list(value_span)} and its unit at {list(unit_span)}. The claim's own "
                "spans are compared by OFFSET, not by the text they happen to spell: two "
                f"different occurrences of {only.value_text!r} are different groundings, and "
                "matching on the string would let one stand in for the other"
            ),
            window_start=window_start,
            window_end=window_end,
            found=rendered,
        )

    if quantity_kind not in only.quantities:
        denotes = ", ".join(sorted(kind.value for kind in only.quantities))
        return StitchRefutation(
            reason=(
                f"this claim declares quantity_kind={quantity_kind.value!r}, but the unit "
                f"{only.unit_text!r} it grounds denotes {denotes} in table "
                f"{table.table_id!r} v{table.version}. The declared kind is checked against "
                "the LOCATED unit rather than against the label, so no label lexicon is "
                "needed and none is consulted"
            ),
            window_start=window_start,
            window_end=window_end,
            found=rendered,
        )

    return None
