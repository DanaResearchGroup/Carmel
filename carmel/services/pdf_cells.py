"""Refuse a claimed PDF region that geometry cannot vouch for. Nothing else.

This module has exactly one public function and it can only ever say NO. It emits no
cell, no locator, no grid, and no row/column index; it never reaches
:mod:`carmel.schemas.datasets`. That is a deliberate scope, arrived at by refutation
rather than by taste, and the history is worth keeping because the obvious design is
the wrong one.

**The design this replaces.** M1 was going to propose a *cell*: a box drawn around some
fragments plus a digest of their text, so that a grouping became positively checkable.
It died on its own motivating example. Draw the box tightly around ``1.0`` and leave the
detached ``/C0`` minus sign -- a separate text-show operation 3.5 pt to its left --
outside it. Every gate the design had passes: nothing straddles the box, no member is
unmapped, the digest matches. And ``-1.0`` is recorded as ``+1.0``, a silent SIGN
INVERSION with nothing downstream looking wrong. The real 8-paper corpus contains that
exact shape 50 times. **A gate that only inspects what you claimed cannot see what you
omitted.**

**Why no threshold rescues it.** The natural repair is to widen the box: treat a
fragment within N points as part of the number. Measured against the corpus, a gap
threshold that keeps a 3.50 pt detached sign attached also merges **69.7%** of genuine
column boundaries, because the sign-to-digit gap (3.50 pt) sits BELOW the median
genuine inter-cell gap (3.97 pt, n=17 641). The two populations are not separable, so
there is no value to tune. Geometry alone cannot decide whether the thing 3.5 pt left
of ``1.0`` is its minus sign or the previous column -- and a module that guesses is
fabricating. So the sign case is resolved by REFUSAL, and this module is that refusal.

**Why "adjacent" is not a segmentation.** An earlier form of this rule segmented the
baseline into runs by a 3 pt gap and inspected the adjacent RUN. That is unsound, and
the reason generalises: MEMBERSHIP IS PRODUCER-CHOSEN. A detached sign 2 pt from the
digit is inside the same run, so it is neither a member (the producer excluded it) nor
the adjacent run (it is in the same one) -- it falls through, and the sign inversion is
back. The rule below instead takes the nearest fragment OUTSIDE the claimed region at
any distance, which has no such crack: whatever was not claimed is a candidate.

**What this costs, and why that number needs a qualifier.** Refusing is not free, and
the price depends on the PROPOSAL, not on this module: on the same corpus the identical
rule refuses 22.11% of tight single-fragment boxes and 7.74% of run-shaped proposals.
The difference is dominated by producers claiming ``3`` out of ``3.14``, which this
layer refuses correctly. The two figures are not even measured over the same
denominator -- a wider band changes what counts as one row, and so changes what a
run-shaped proposal IS. **A refusal rate may never be quoted without the proposal shape
it was measured on**, and no producer exists yet, so this module has no single cost.
Re-measure against the real proposal shape when one does.

**Why there are two entry points.** :func:`refuse_region` answers the operational
question -- may this region be vouched for -- and one reason is enough to say no. But a
first-wins reason makes a TALLY over many regions lie: check order, not the corpus,
decides which category a region lands in, so ``STRADDLED`` looks rare simply because
``UNMAPPED_MEMBER`` is tested earlier. :func:`region_refusals` exists so that a
measurement can count what is actually there, and so that any consumer measuring this
layer measures THIS layer rather than a re-implementation of it.

**What ``None`` means.** Only that no reason to refuse was found. It is NOT an
approval, NOT evidence, and NOT a verification. Nothing here may be persisted as a
positive artifact -- there is deliberately no ``RegionCheck``, no ``clean``, no
``accepted`` and no ``not_refused`` value in this module, because a stored
"passed the refusal layer" record would launder into exactly the admissive evidence
that :class:`carmel.services.dataset_replay.RefutationStatus` already had to be walled
off from. Persist refusals; persist nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from carmel.services.numeric import AffixClass, classify_abutting_affix
from carmel.services.pdf_fragments import FragmentExtraction, GlyphMapping, TextFragment

#: How far from the claimed baseline a fragment may sit and still count as being on the
#: same line, in page-space units.
#:
#: **This is a declared scope limit, not geometric truth**, and it is the only tuned
#: parameter in this module. The substrate offers baselines and rendered heights, never
#: glyph boxes, so vertical extent is necessarily inferred and no threshold-free
#: formulation exists. Stating it as a policy constant is the honest alternative to
#: hiding it inside a comparison.
#:
#: Set from the corpus, and NOT from the reasoning that first suggested it. The value
#: here was 0.75 pt, justified as "wide enough for sub-point baseline jitter, narrow
#: enough that the line above stays out". The second half of that was simply false:
#: measured line spacing on this corpus is **10.4-11.5 pt**, so nothing at 2 pt was
#: ever the next row, and the band was excluding same-line content for a danger that
#: does not exist there.
#:
#: What the baselines actually do (7676 gaps between distinct baselines on a page):
#: 38.4% fall in [0, 2) pt, then a shallow trough of ~28% spread across [2, 8), then
#: 28.8% in [8, 12) which is the real line spacing. **There is no valley to cut at.**
#: The same-line and next-line populations overlap through the middle exactly as the
#: detached-sign gap and the genuine column gap do horizontally -- the same structural
#: result, one axis over.
#:
#: So the band is chosen on refusal behaviour instead (5332 single-fragment numeric
#: proposals):
#:
#: ========  =======  ===============
#: band      refused  sign-class hits
#: ========  =======  ===============
#: 0.0 pt     14.82%              265
#: 0.75 pt    15.44%              284
#: 2.0 pt     21.32%              475
#: 4.0 pt     22.11%              489
#: 6.0 pt     22.34%              490
#: 9.0 pt     22.02%              476
#: ========  =======  ===============
#:
#: The step from 0.75 to 2.0 admits **191 additional sign-class hits** -- detached
#: signs that the narrow band was silently missing, which is the precise failure this
#: module exists to prevent. Past 2.0 the curve saturates: 4.0 buys 14 more hits for
#: 0.79 pp, 6.0 buys one more, and 9.0 goes DOWN, because a band that wide starts
#: pulling in a nearer safe neighbour that displaces a dangerous farther one.
#:
#: 4.0 pt sits inside that plateau rather than on its edge, and it is deliberately wide
#: enough to reach the 3-5 pt superscript zone, so a sign or exponent set as a
#: superscript is inspected instead of being declared out of scope. Under-refusing is
#: the dangerous direction here; a refusal costs a datum, a missed sign costs its sign.
#:
#: **This remains a declared scope limit, not geometric truth.** The substrate offers
#: baselines and rendered heights, never glyph boxes, so vertical extent is necessarily
#: inferred and no threshold-free formulation exists. A font-RELATIVE policy was tried
#: and refuted: it behaves like a fixed ~4.8 pt band, and the field it depended on
#: turned out to be a corpus-wide constant (see ``TextFragment.font_height``).
BASELINE_BAND = 4.0

#: How far above the claimed baseline a neighbour must sit before it is read as a
#: SUPERSCRIPT rather than as same-line text.
#:
#: Only ever widens what refuses, never narrows it, so it cannot hide a dangerous
#: neighbour: above this rise, a leading sign on the right stops being read as the next
#: cell's own sign and starts being read as this value's exponent. Set at the top of the
#: measured same-line jitter cluster -- 38.4% of baseline gaps fall in [0, 2) pt, which
#: is ordinary within-line variation, while a real superscript in this corpus sits
#: 3-5 pt up.
_SUPERSCRIPT_RISE = 2.0


class RegionRefusalReason(StrEnum):
    """Why a claimed region cannot be vouched for.

    A SIBLING of the two enums it must not be confused with, and separate from both on
    purpose:

    * :class:`carmel.services.dataset_replay.RefutationStatus` is about attempted
      falsification of a claim and interacts with replay outcome semantics; its
      ``NOT_REFUTED`` famously does not mean verified. A refusal here is not a
      falsification attempt at all.
    * :class:`carmel.schemas.datasets.UnextractedReason` means "a LOCATED condition
      statement did not become a claim" and has no region, glyph or layout members.

    Reusing either would make a geometry decision readable as an evidentiary one.
    """

    EXTRACTION_UNAVAILABLE = "extraction_unavailable"
    """The fragment lane produced nothing usable -- pypdf absent, the capability check
    refused, the engine contradicted that check mid-walk, or the reader walk failed.
    NOTHING about this document may be claimed, so the region cannot be vouched for
    regardless of its geometry.

    Which of those four it was lives on
    :class:`carmel.services.pdf_fragments.FragmentAvailability`, deliberately not
    here: it is a property of the DOCUMENT, uniform across every region in it, and a
    per-region enum is the wrong place to keep a per-document fact."""

    PAGE_INCOMPLETE = "page_incomplete"
    """This region's page failed, could not be inspected, or the document was truncated
    before finishing. The dangerous neighbour that would have refused this region may
    simply never have been extracted, so a clean result here would be an artefact of
    the loss. Fails toward refusal, which is the whole point of recording loss per-page
    rather than as a bare ``lossy`` flag."""

    EMPTY = "empty"
    """No fragment lies inside the claimed region. A region with no members cannot be
    evidence of anything, and silently returning "no problem found" for an empty box is
    the most direct way to manufacture a groundless claim."""

    UNMAPPED_MEMBER = "unmapped_member"
    """A member fragment contains a glyph that never decoded -- ``/C0``, ``(cid:3)``,
    U+FFFD. Its text is a MARKER, not data, and this is the case that finally makes
    :class:`carmel.services.pdf_fragments.GlyphMapping` load-bearing."""

    ROTATED_MEMBER = "rotated_member"
    """A member is rotated with respect to the page. Its x-extent is not a horizontal
    extent, so the whole left/right analysis below is meaningless for it."""

    ROTATED_NEIGHBOUR = "rotated_neighbour"
    """A rotated fragment shares the band. Same reason as above, one step out: its
    geometry cannot be compared against the region's, so neither its exclusion nor its
    inclusion can be justified. 74 of 5332 corpus proposals have one."""

    ADJACENT_UNREADABLE = "adjacent_unreadable"
    """The nearest non-blank fragment outside the region is a numeral-modifying affix
    (see :func:`carmel.services.numeric.classify_affix`). This is the detached-sign
    case, and the reason this module exists."""

    INVALID_GEOMETRY = "invalid_geometry"
    """The claimed box is not a box: a non-finite coordinate, or an x-extent that does
    not increase. Every comparison below is a float ``<``, and NaN makes all of them
    quietly False -- so an unchecked NaN box passes every refusal and returns ``None``,
    which is the most permissive answer available. Also catches a fragment whose own
    extent runs backwards, which mirrored or negative-scale text can produce while
    ``rotated`` is still False."""

    FOREIGN_MEMBER = "foreign_member"
    """A claimed member is not one of the fragments in this extraction. Nothing further
    can be checked about it -- it has no page, no neighbours and no provenance here --
    and a region built from fabricated members would otherwise satisfy every test below
    while describing text the document never contained."""

    MEMBER_OUTSIDE_REGION = "member_outside_region"
    """A claimed member lies outside the claimed box, or off its page or band. This is
    the exclusion attack inverted: naming the detached ``−`` as a MEMBER while drawing
    the box around ``1.0`` removes it from the outsiders, so the adjacency check never
    sees it and the sign is lost with no refusal raised."""

    UNCOMPARABLE_NEIGHBOUR = "uncomparable_neighbour"
    """A NON-MEMBER fragment on this page has geometry that cannot be compared against
    the region's: a non-finite baseline, or an extent that is non-finite or backwards.

    The sibling of :attr:`INVALID_GEOMETRY`, which states this same rule and applies it
    to the claimed box and its members only. Leaving the outsiders unchecked left the
    hole open on the side the members do not cover, and it is silent in TWO different
    ways at once:

    * a NaN or backwards X-EXTENT survives into the band and then vanishes from every
      comparison, because ``<`` against NaN is quietly False -- so it is neither
      recorded as a straddler nor found as the nearest neighbour, and the region reads
      clean because the check that would have refused it could not run;
    * a NaN BASELINE never reaches the band at all, because the band test is itself a
      ``<=`` that NaN fails. It is dropped one step earlier, for a different reason,
      which is why this check is scoped to the PAGE and not to the band -- a band-scoped
      check would inspect only the fragments that already passed the test NaN defeats.

    Neither the exclusion nor the inclusion of such a fragment can be justified, exactly
    as for :attr:`ROTATED_NEIGHBOUR`, so this halts rather than joining an independent
    tally. **Zero corpus proposals reach it**: the 8-paper corpus holds no non-finite
    and no backwards fragment extent. It guards the shape of the input, not an observed
    one -- which is the same footing :attr:`INVALID_GEOMETRY` has always stood on."""

    STRADDLED = "straddled"
    """A fragment that is NOT a member overlaps the claimed region horizontally. The
    region's own boundary is therefore not clean: something was cut through rather than
    included or excluded. Costs 0.00% on run-shaped proposals (nothing overlaps, by
    construction) and 13.95% on tight single-fragment boxes -- where it is largely
    CORRECT, since a box drawn around the ``3`` of ``3.14`` really does cut one."""


#: Reasons after which nothing further about the region can be EVALUATED, so
#: :func:`region_refusals` stops there and its result is a truncated census.
#:
#: Membership is a claim about evaluability, never about severity. A halting reason has
#: destroyed the input that the remaining checks read: an unavailable or incomplete
#: extraction means an absent neighbour proves nothing; a non-finite extent makes every
#: ``<`` below quietly False; a foreign or out-of-box member means the members tuple no
#: longer partitions the page, so the outsider set that every band check reads is
#: computed from a claim already known to be false.
#:
#: Everything absent from this set is an INDEPENDENT finding, and the reason this
#: constant exists: an unmapped member no longer hides a straddler, and a straddler no
#: longer hides the detached sign next to it.
#:
#: ``ROTATED_NEIGHBOUR`` halts, and that one is a genuine limit rather than a
#: convenience. A rotated fragment's ``x_start``/``x_end`` do not bound it horizontally,
#: so it can be neither ruled out as the nearest neighbour nor recorded as a straddler
#: -- recording it as one would be inventing a finding from a meaningless extent. **A
#: tally of ``STRADDLED`` is therefore a lower bound**, short by at most the regions
#: that halt here: 74 of 5332 (1.4%) on the 8-paper corpus.
HALTS_EVALUATION: frozenset[RegionRefusalReason] = frozenset(
    {
        RegionRefusalReason.EXTRACTION_UNAVAILABLE,
        RegionRefusalReason.PAGE_INCOMPLETE,
        RegionRefusalReason.INVALID_GEOMETRY,
        RegionRefusalReason.EMPTY,
        RegionRefusalReason.FOREIGN_MEMBER,
        RegionRefusalReason.MEMBER_OUTSIDE_REGION,
        RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR,
        RegionRefusalReason.ROTATED_NEIGHBOUR,
    }
)


@dataclass(frozen=True)
class ClaimedRegion:
    """The box a producer drew, and the fragments it says are inside it.

    ``members`` is given explicitly rather than derived from the box, because the
    producer's CHOICE is the thing under examination. Deriving membership here would
    make this module check its own arithmetic instead of the producer's claim, and the
    exclusion attack in this module's docstring is precisely a producer excluding a
    fragment that geometry would have included.

    **But an unchecked claim is a second way in, and it is the same way in.** Every
    field here is caller-controlled, so each one is an attack surface that
    :func:`refuse_region` must close before it trusts any of them:

    * a ``baseline_y`` set 1 pt off the members' real baseline moves the band away from
      the detached sign, and the region comes back clean;
    * a ``members`` tuple holding a fragment that is not in the extraction at all
      fabricates content that no page ever carried;
    * naming the detached ``−`` as a MEMBER while drawing the box around ``1.0`` gets
      it filtered out of the outsiders and silences the very refusal it should trigger.

    The last one is the original exclusion attack wearing a different hat: the producer
    still ends up with ``1.0`` and no complaint, having *included* the sign rather than
    omitted it. Membership is therefore a claim to be checked against the box, not a
    fact to be taken.
    """

    page: int
    x_start: float
    x_end: float
    baseline_y: float
    members: tuple[TextFragment, ...]


@dataclass(frozen=True)
class RegionRefusal:
    """A refusal, with enough detail for an operator to see what was refused and why."""

    reason: RegionRefusalReason
    detail: str
    """Short, human-readable. Never the document's text beyond the offending token."""

    affix: AffixClass | None = None
    """Set only for :attr:`RegionRefusalReason.ADJACENT_UNREADABLE`, and not always
    even then -- an unmapped neighbour refuses without being classified."""


def _is_sane_extent(start: float, end: float) -> bool:
    """A FRAGMENT's horizontal extent: finite, and running the way page space runs.

    Both halves are load-bearing. NaN makes every ``<`` comparison in this module
    quietly False, so an unchecked NaN box would satisfy the straddle test, find no
    neighbours, and return ``None`` -- the most permissive answer -- without a single
    predicate having actually run. And an extent running backwards is not hypothetical:
    mirrored or negative-scale text yields ``x_end < x_start`` while ``rotated`` is
    still False, which inverts left, right and overlap all at once.

    ZERO WIDTH IS ADMITTED HERE, DELIBERATELY, and this is the one place the fragment
    rule and the region rule part company. A combining diacritic draws over the glyph
    before it and does not advance the pen, so ``x_start == x_end`` is what a correct
    extractor reports for it. Measured on the eight-paper corpus: 149 of 78,178
    fragments have zero width, and 23 of those are ``MAPPED`` -- ``´`` and ``¨``, real
    accents on real words, spread across 7 of the 8 papers. Refusing them would refuse
    every region containing an accented character, which is why this predicate is
    ``<=`` and not ``<``. The remaining 126 are ``UNMAPPED`` (``/C24``-style markers),
    already carried by :attr:`GlyphMapping` rather than by geometry.
    """
    return math.isfinite(start) and math.isfinite(end) and start <= end


def _is_sane_region_extent(start: float, end: float) -> bool:
    """A CLAIMED REGION's horizontal extent, which must be strictly positive.

    A region is an area someone claims a cell occupies, not a glyph, so the combining
    -mark exemption above does not apply to it: a box of zero width contains nothing,
    can straddle nothing, and would sail through the neighbour search to ``None`` --
    the same permissive non-answer the NaN check exists to stop. The refusal text said
    "non-increasing" while the predicate accepted equality, so the gate was looser than
    its own message for as long as both call sites shared one function.
    """
    return math.isfinite(start) and math.isfinite(end) and start < end


def _edge_token(text: str, *, from_end: bool) -> str:
    """The trailing or leading whitespace-delimited token of a fragment's text.

    Classifying the WHOLE fragment string is wrong in both directions, because a
    fragment is a text-show operation and not a token: ``'Ref. [23] -'`` ends in a
    dangerous affix while the whole string is not one, and a fragment holding a whole
    label would never match an affix exactly.
    """
    parts = text.split()
    if not parts:
        return ""
    return parts[-1] if from_end else parts[0]


def refuse_region(extraction: FragmentExtraction, region: ClaimedRegion) -> RegionRefusal | None:
    """Refuse ``region``, or return ``None`` if no reason to refuse was found.

    One reason is all it takes to refuse, so this returns the first one found and stops
    mattering there. **Do not tally these** -- the category you get is decided by the
    order of the checks in :func:`region_refusals`, not by the region. Count that
    function's full result instead.

    Read :attr:`RegionRefusalReason` and this module's docstring before treating a
    ``None`` here as anything at all.
    """
    refusals = region_refusals(extraction, region)
    return refusals[0] if refusals else None


def region_refusals(extraction: FragmentExtraction, region: ClaimedRegion) -> tuple[RegionRefusal, ...]:
    """Every reason to refuse ``region`` that could be evaluated, in check order.

    Empty means no reason to refuse was found -- see this module's docstring for what
    that does and does not mean. It is the same non-answer as :func:`refuse_region`
    returning ``None``, wearing a different type.

    Takes the WHOLE :class:`~carmel.services.pdf_fragments.FragmentExtraction` rather
    than one page's fragments, because the page-local view cannot see ``available``,
    ``page_failures`` or ``truncated`` -- and would therefore find nothing, the most
    permissive answer, exactly when the evidence is known to be partial.

    **The result is a census only up to its last element.** If that element's reason is
    in :data:`HALTS_EVALUATION`, the checks after it never ran and their absence says
    nothing about the region. Any consumer counting reasons across a corpus must count
    the halts too, or it will report the remaining categories as rarer than they are.

    At most one refusal per reason: a region with three unmapped members has one
    ``UNMAPPED_MEMBER``, because the unit being counted is the region.
    """
    found: dict[RegionRefusalReason, RegionRefusal] = {}

    def note(reason: RegionRefusalReason, detail: str, affix: AffixClass | None = None) -> tuple[RegionRefusal, ...]:
        found.setdefault(reason, RegionRefusal(reason, detail, affix))
        return tuple(found.values())

    if not extraction.available:
        # ONE reason for all four unavailability states, on purpose. The region-level
        # answer is identically "nothing about this document was established", and a
        # per-DOCUMENT fault does not belong in a per-REGION census: splitting it here
        # would multiply the categories a corpus tally has to carry while saying
        # nothing about any region. The state is named in the detail for whoever reads
        # one refusal, and the detail is NOT the carrier -- anything that needs the
        # distinction structurally reads `extraction.status`, which every caller
        # already holds, because it passed the extraction in.
        return note(
            RegionRefusalReason.EXTRACTION_UNAVAILABLE,
            f"the fragment lane is unavailable for this document ({extraction.status})",
        )

    # `lossy` without either carrier is loss this module cannot LOCATE, so it cannot
    # tell whether this page kept everything. Checked separately from the page-specific
    # test below, which stays precise on purpose: recording failures per page is what
    # lets a clean page next to a broken one still be answerable.
    unlocatable_loss = extraction.lossy and not (extraction.truncated or extraction.page_failures)
    if (
        extraction.truncated
        or unlocatable_loss
        or any(failure.page == region.page for failure in extraction.page_failures)
    ):
        return note(
            RegionRefusalReason.PAGE_INCOMPLETE,
            f"page {region.page} is incomplete, so an absent neighbour proves nothing",
        )

    if not _is_sane_region_extent(region.x_start, region.x_end) or not math.isfinite(region.baseline_y):
        return note(
            RegionRefusalReason.INVALID_GEOMETRY,
            "the claimed box has a non-finite, backwards or zero-width extent",
        )

    if not region.members:
        return note(RegionRefusalReason.EMPTY, "the region claims no fragments")

    # Identity, not equality: two fragments with the same text and geometry are still
    # different pieces of evidence, and the question here is whether THIS object is one
    # the extraction produced -- which is also what makes the `id()` membership test
    # below sound, since every member is now known to be a live element of that tuple.
    extraction_ids = {id(fragment) for fragment in extraction.fragments}

    for member in region.members:
        if id(member) not in extraction_ids:
            return note(
                RegionRefusalReason.FOREIGN_MEMBER,
                "a claimed member is not a fragment of this extraction",
            )
        if not _is_sane_extent(member.x_start, member.x_end) or not math.isfinite(member.baseline_y):
            return note(
                RegionRefusalReason.INVALID_GEOMETRY,
                "a member fragment has a non-finite or backwards extent",
            )
        if member.page != region.page:
            return note(
                RegionRefusalReason.MEMBER_OUTSIDE_REGION,
                "a claimed member lies on another page",
            )
        if member.glyph_mapping is GlyphMapping.UNMAPPED:
            note(
                RegionRefusalReason.UNMAPPED_MEMBER,
                "a member fragment carries an unmapped-glyph marker, not text",
            )
        if member.rotated:
            # Its extent is not a horizontal one, so the box test below would compare
            # the region against a number that does not mean what the comparison
            # assumes. Recording the rotation and declining to place it is the honest
            # move; declaring it outside the box would be inventing a second finding
            # out of the first one's uncertainty.
            note(
                RegionRefusalReason.ROTATED_MEMBER,
                "a member fragment is rotated, so its horizontal extent is not one",
            )
            continue
        if (
            member.x_start < region.x_start
            or member.x_end > region.x_end
            or abs(member.baseline_y - region.baseline_y) > BASELINE_BAND
        ):
            return note(
                RegionRefusalReason.MEMBER_OUTSIDE_REGION,
                "a claimed member lies outside the claimed box",
            )

    member_ids = {id(member) for member in region.members}
    others = [
        fragment
        for fragment in extraction.fragments
        if fragment.page == region.page and id(fragment) not in member_ids and fragment.text.strip()
    ]

    # Checked over the PAGE, before the band is cut, and this order is the whole point.
    # The band test below is `abs(baseline_y - ...) <= BASELINE_BAND`, which a NaN
    # baseline fails -- so a fragment with one is already gone by the time a band-scoped
    # check could look at it, and such a check would only ever inspect fragments that
    # passed the very comparison NaN defeats. Blank fragments are excluded above for the
    # same reason the neighbour scan skips them: they are never a straddler and never a
    # neighbour, so their geometry is not read and cannot mislead.
    if any(
        not math.isfinite(fragment.baseline_y) or not _is_sane_extent(fragment.x_start, fragment.x_end)
        for fragment in others
    ):
        return note(
            RegionRefusalReason.UNCOMPARABLE_NEIGHBOUR,
            "a non-member fragment on this page has a non-finite or backwards extent",
        )

    outsiders = [fragment for fragment in others if abs(fragment.baseline_y - region.baseline_y) <= BASELINE_BAND]

    if any(fragment.rotated for fragment in outsiders):
        return note(
            RegionRefusalReason.ROTATED_NEIGHBOUR,
            "a rotated fragment shares the band; its geometry is not comparable",
        )

    if any(fragment.x_end > region.x_start and fragment.x_start < region.x_end for fragment in outsiders):
        note(
            RegionRefusalReason.STRADDLED,
            "a non-member fragment overlaps the region, so its boundary is not clean",
        )

    # Nearest NON-BLANK neighbour on each side, at any distance. Blank fragments are
    # skipped rather than treated as neighbours: a bare single-space text-show
    # operation between a detached sign and its digit would otherwise SHIELD the sign
    # from this check, which it does for 20 of 5332 corpus proposals.
    left = max(
        (fragment for fragment in outsiders if fragment.x_end <= region.x_start),
        key=lambda fragment: fragment.x_end,
        default=None,
    )
    right = min(
        (fragment for fragment in outsiders if fragment.x_start >= region.x_end),
        key=lambda fragment: fragment.x_start,
        default=None,
    )

    for neighbour, from_end, side in ((left, True, "left"), (right, False, "right")):
        if neighbour is None:
            continue
        # An unmapped neighbour refuses on the FLAG, before any token matching. Its
        # text is markers, so an edge token like `n=/C0` matches no affix while the
        # fragment is exactly the unreadable thing next to the number that this module
        # exists to catch. Asking the classifier first would make the check depend on
        # the marker landing alone in its own fragment.
        if neighbour.glyph_mapping is GlyphMapping.UNMAPPED:
            return note(
                RegionRefusalReason.ADJACENT_UNREADABLE,
                f"the nearest fragment to the {side} carries an unmapped-glyph marker",
            )
        # A RAISED neighbour is a superscript, and a superscript binds to what precedes
        # it -- so `−3` above and right of `1.0` is its exponent, not the next cell's
        # negative value. Only the geometry can tell those apart, so the direction rule
        # in `classify_abutting_affix` is overridden here rather than guessed there.
        affix = classify_abutting_affix(
            _edge_token(neighbour.text, from_end=from_end),
            from_end=from_end,
            always_reaches=neighbour.baseline_y > region.baseline_y + _SUPERSCRIPT_RISE,
        )
        if affix is not None:
            return note(
                RegionRefusalReason.ADJACENT_UNREADABLE,
                f"the nearest fragment to the {side} is a {affix.value} affix",
                affix=affix,
            )

    return tuple(found.values())
