"""Produce a validated :class:`ConditionSetEnvelope` from one stored artifact.

Carmel could already STORE, LOAD and REPLAY a condition set before this module
existed -- :mod:`carmel.services.condition_set_bridge` and
:func:`carmel.services.dataset_replay.replay_condition_set` were both complete --
but nothing could MAKE one. Every condition-set envelope in the suite was
hand-built in test code, which meant the replayer had never once been handed real
producer output. This module closes that gap.

WHAT THIS PRODUCER IS FOR, AND WHY IT IS SHAPED THIS WAY

A char span into extracted running text can LOCATE a scalar statement -- "the
mixture was preheated to 323 K". It cannot locate a SERIES data point, because a
series is a structure (rows against columns, or points against axes) and running
text carries no such structure at all. That asymmetry is why this module exists:
it is the GROUNDED destination for the scalar-shaped half of what a text-only
extractor can see, so that the series-shaped half can be refused elsewhere
without also destroying this one.

Be exact about what "grounded destination" does and does not mean, because the
tempting phrasing -- "the honest destination for prose-local scalars" -- is an
overclaim this module cannot back. This code CANNOT prove prose-locality, and it
cannot prove that a located label, value and unit are predicated of one another
by the paper. It proves that each quote is where the locator says it is, and it
derives the numeric/unit normalization deterministically from the value quote.
Everything past that is the caller's assertion, recorded unverified.

One shape of that gap is now CLOSED, and the closure is narrow enough to state
exactly. A caller used to be able to stitch the label "pressure" to the value
"823" and the unit "atm" out of a sentence that says 823 K and 1.2 atm: every
span grounded, replay reported VERIFIED, and the paper never stated that
condition. Co-location could not close it -- that false triple comes from a
single sentence -- so the rule is uniqueness instead: the span COVERING a
claim's three grounds must hold exactly one number+unit construct, that
construct must be the claimed value and unit compared by offset, and its unit
must denote the declared quantity. See :mod:`carmel.services.stitching`; the
same gate re-runs in
:func:`carmel.services.dataset_replay.replay_condition_set`, because a
write-path-only gate says nothing about an envelope built by another route.

That gate REFUTES; it never verifies. A claim surviving it is not thereby
shown to be what the paper predicates -- only that one named refutation was
attempted and did not fire. Known shapes it does NOT refuse: a one-sided bound
or method threshold ("above 60 cm/s") reads as a single construct, and shared
dimensionless spellings cannot separate mole fraction from equivalence ratio
from a relative uncertainty.

GROUNDING PROVES LOCATION, NEVER MEANING. Every ``SourceRef`` this producer emits
is independently verified to be an exact, located substring of the authenticated
document. NOTHING here verifies that the located string MEANS what the caller
says it means -- that a quote the caller labelled "initial temperature" really is
the initial temperature, or that a number in the text is a reported condition
rather than a chart tick. The schema records the caller's assertion; it does not
bless it.

THE THREE-WAY SPLIT IS THE HONESTY MECHANISM

A condition either resolves to one grounded number (:class:`ScalarConditionSpec`),
or to one grounded categorical token (:class:`CategoricalConditionSpec`), or it is
REFUSED with the reason recorded and the span still grounded
(:class:`UnextractedConditionSpec`). A sweep, a range, a one-sided bound or a
qualitative-only statement is not squeezed into a single number -- it is recorded
as an explicit refusal that still points at the text it refused. A refusal that
names its own span is evidence; a silently dropped condition is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    CaptionLabelKey,
    ConditionAttribution,
    ConditionSetEnvelope,
    DeviceClassDeclaration,
    EmbeddedTableInventory,
    GroundedCategoricalClaim,
    GroundedScalarClaim,
    MemberSheetKey,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    SubjectRefusalReason,
    TableCellLocator,
    UnextractedConditionStatement,
    UnextractedReason,
    UnresolvedSubject,
)
from carmel.services import units
from carmel.services.dataset_producer import (
    _ACTIVE,
    _ROOT_NODE_ID,
    DatasetProducerError,
    _measured_value,
    _prepare_grounding,
    ground_quote,
)
from carmel.services.numeric import GlyphHealth, QuoteRole, SourceContext
from carmel.services.pdf_fragments import GlyphRepair
from carmel.services.stitching import (
    StitchGateUnrunnable,
    StitchRefutation,
    refute_stitched_claim,
)

__all__ = [
    "CategoricalConditionSpec",
    "ConditionSetProducerError",
    "DeviceClassSpec",
    "ScalarConditionSpec",
    "TableCellGrounding",
    "UnextractedConditionSpec",
    "UnresolvedSubjectSpec",
    "produce_condition_set_from_artifact",
]


class ConditionSetProducerError(DatasetProducerError):
    """A condition set could not be honestly produced.

    Subclasses :class:`DatasetProducerError` deliberately: callers that already
    fail closed on "a producer refused" keep working unchanged, while a caller
    that wants to tell the two producers apart still can.
    """


def _require_int_occurrences(owner: str, **occurrences: int | None) -> None:
    """Reject non-int occurrence values, ``bool`` included.

    These specs are frozen plain dataclasses, so nothing downstream re-checks
    their fields. ``bool`` is a subclass of ``int`` in Python, so a bare
    ``isinstance(x, int)`` would silently accept ``True``/``False`` as
    occurrence 1/0 -- almost certainly a caller typo, never a real
    disambiguation intent. Mirrors ``MeasurementSpec.__post_init__``.
    """
    for name, value in occurrences.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConditionSetProducerError(
                f"{owner}.{name}={value!r} must be an int or None, not {type(value).__name__} "
                "-- bool is a subclass of int in Python and would silently mean occurrence 0/1"
            )


@dataclass(frozen=True, slots=True)
class TableCellGrounding:
    """Ground ONE quote at a specific cell of a specific PDF table grid.

    This is the input nothing in the codebase supplied before: it names WHICH
    table (``table_key``), WHERE in it (``row``/``col``), and the extracted
    inventory that DEFINES that grid (``inventory``). Given one, the producer
    builds the existing :class:`~carmel.schemas.datasets.TableCellLocator` for the
    quote instead of a :class:`~carmel.schemas.datasets.CharSpanLocator` -- it is
    NOT a new locator, it is the missing path to the tiny ordinal-only one the
    schema already models.

    ``table_key`` is the decision this ticket had to make: no existing spec field
    carried it. It is the schema's own discriminated union, but only ONE of its two
    arms is reachable today: a :class:`~carmel.schemas.datasets.CaptionLabelKey`
    (the printed caption, e.g. ``"Table 1"``) over a ``PAPER_PDF`` node. The union's
    other arm, :class:`~carmel.schemas.datasets.MemberSheetKey` (a workbook sheet
    name), is RESERVED and NOT YET REACHABLE: :meth:`_CellCiter.validate` refuses any
    grounding whose root node is not ``PAPER_PDF``, and a sheet is not a PDF node, so
    a ``MemberSheetKey`` grounding can only ever be rejected -- do not supply one
    until that guard is taught to accept a sheet's cells. A key is needed at all
    because ``row``/``col`` alone are meaningless without saying which of a node's
    several tables they index into. It is carried on THIS object, alongside the row
    and column, rather than once per spec, because a single claim can in principle
    draw its label and its value from two different tables of one document.

    ``inventory`` is embedded, not looked up: the producer places every cited
    inventory into the envelope's ``table_inventories`` so the citation resolves
    from the envelope's own bytes, never from an evidence store at replay time.
    Its ``inventory_sha256`` becomes the locator's ``pdf_table_inventory_sha256``
    -- which is why the producer NEVER emits an ``Absent`` sha for a cell it
    grounds: an absent sha is a citation nothing can ever resolve.

    The grounded quote STRING still lives on the spec (``value_quote`` etc.): a
    cell citation carries no text of its own, so the spec's quote is what the
    producer holds the cell text to. The two must be EXACTLY equal -- whole cell
    text against whole quote -- or the producer refuses (the settled matching
    contract), because "the cell contains my text" would let ``8`` cite a cell
    reading ``1-8``.
    """

    table_key: CaptionLabelKey | MemberSheetKey
    row: int
    col: int
    inventory: EmbeddedTableInventory

    def __post_init__(self) -> None:
        if not isinstance(self.table_key, (CaptionLabelKey, MemberSheetKey)):
            raise ConditionSetProducerError(
                f"TableCellGrounding.table_key={self.table_key!r} must be a CaptionLabelKey or "
                f"MemberSheetKey, not {type(self.table_key).__name__}"
            )
        if not isinstance(self.inventory, EmbeddedTableInventory):
            raise ConditionSetProducerError(
                f"TableCellGrounding.inventory must be an EmbeddedTableInventory, not "
                f"{type(self.inventory).__name__} -- the producer embeds it and reads the cell's own "
                "text from it, so a stand-in that only carries a sha would defeat the exact-equality check"
            )
        for name, value in (("row", self.row), ("col", self.col)):
            # bool is an int subclass; a `True` row is a caller typo, never ordinal 1.
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConditionSetProducerError(
                    f"TableCellGrounding.{name}={value!r} must be a non-negative int -- a negative or "
                    "non-integer ordinal locates no real table cell"
                )


def _reject_cell_with_occurrence(owner: str, **quotes: tuple[int | None, TableCellGrounding | None]) -> None:
    """Refuse a quote that is BOTH text-disambiguated and cell-grounded.

    An ``occurrence`` disambiguates a substring SEARCH of running text; a
    ``TableCellGrounding`` says the quote is not in running text at all but at a
    named grid cell. Supplying both is a contradiction the producer cannot honour,
    and silently preferring one would ground the quote somewhere the caller did not
    unambiguously ask for.
    """
    for name, (occurrence, cell) in quotes.items():
        if occurrence is not None and cell is not None:
            raise ConditionSetProducerError(
                f"{owner}.{name}: an occurrence ({occurrence!r}) disambiguates a running-text search "
                "while a TableCellGrounding names a table cell -- a quote cannot be grounded both ways, "
                "so supply exactly one"
            )


@dataclass(frozen=True, slots=True)
class ScalarConditionSpec:
    """One stated condition that resolves to a single grounded number.

    Carries no ``axis_id`` and no ``AxisRole``: a condition is not a point on a
    series. It satisfies ``dataset_producer._ValueQuoteSpec`` structurally, which
    is what lets it share :func:`_measured_value` with the dataset producer
    without either borrowing the other's vocabulary.
    """

    claim_id: str
    label_quote: str
    quantity_kind: units.QuantityKind
    value_quote: str
    unit_quote: str
    label_occurrence: int | None = None
    value_occurrence: int | None = None
    unit_occurrence: int | None = None
    #: Per-quote cell grounding. When set, that quote is located at a table cell
    #: (a TableCellLocator) instead of searched in running text (a CharSpanLocator).
    label_cell: TableCellGrounding | None = None
    value_cell: TableCellGrounding | None = None
    unit_cell: TableCellGrounding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity_kind, units.QuantityKind):
            raise ConditionSetProducerError(
                f"ScalarConditionSpec.quantity_kind={self.quantity_kind!r} must be a genuine "
                f"QuantityKind member, not {type(self.quantity_kind).__name__} -- QuantityKind "
                "is a StrEnum, so a plain string equal to a member's value would compare `==` "
                "equal without actually being that member"
            )
        _require_int_occurrences(
            "ScalarConditionSpec",
            label_occurrence=self.label_occurrence,
            value_occurrence=self.value_occurrence,
            unit_occurrence=self.unit_occurrence,
        )
        _reject_cell_with_occurrence(
            "ScalarConditionSpec",
            label=(self.label_occurrence, self.label_cell),
            value=(self.value_occurrence, self.value_cell),
            unit=(self.unit_occurrence, self.unit_cell),
        )


@dataclass(frozen=True, slots=True)
class CategoricalConditionSpec:
    """One stated condition whose value is a token, not a number.

    Fuel identity, diluent, reactor material: things a paper states as a word.
    There is no unit and no numeric normalization, so this deliberately does NOT
    go through :func:`_measured_value` -- the token is grounded and recorded raw.
    """

    claim_id: str
    label_quote: str
    token_quote: str
    label_occurrence: int | None = None
    token_occurrence: int | None = None
    label_cell: TableCellGrounding | None = None
    token_cell: TableCellGrounding | None = None

    def __post_init__(self) -> None:
        _require_int_occurrences(
            "CategoricalConditionSpec",
            label_occurrence=self.label_occurrence,
            token_occurrence=self.token_occurrence,
        )
        _reject_cell_with_occurrence(
            "CategoricalConditionSpec",
            label=(self.label_occurrence, self.label_cell),
            token=(self.token_occurrence, self.token_cell),
        )


@dataclass(frozen=True, slots=True)
class UnextractedConditionSpec:
    """A condition the extractor REFUSES to reduce to one value, span recorded.

    ``quantity_kind`` is a ``Maybe``: a refused statement may still be known to be
    a temperature even when no single temperature can be stated. It carries no
    unit, which is why the ref-less obligation machinery deliberately does not
    ask this class for a ``quantity_kind`` claim -- that rule is a predicate over
    ``unit_raw``, and this class has no unit.
    """

    statement_id: str
    label_quote: str
    statement_quote: str
    reason: UnextractedReason
    quantity_kind: units.QuantityKind | None = None
    label_occurrence: int | None = None
    statement_occurrence: int | None = None
    #: BOTH refs of an unextracted statement -- its label and the statement itself
    #: -- can be cell-grounded, so a refused range still points at the cells it
    #: refused rather than declining to say where it declined.
    label_cell: TableCellGrounding | None = None
    statement_cell: TableCellGrounding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, UnextractedReason):
            raise ConditionSetProducerError(
                f"UnextractedConditionSpec.reason={self.reason!r} must be a genuine "
                f"UnextractedReason member, not {type(self.reason).__name__}"
            )
        if self.quantity_kind is not None and not isinstance(self.quantity_kind, units.QuantityKind):
            # The same StrEnum trap the other spec fields close. Left open here,
            # a bare "temperature" would be coerced downstream by pydantic and
            # the refusal would surface as a schema ValidationError from deep
            # inside construction rather than as a producer refusal naming the
            # field the caller got wrong.
            raise ConditionSetProducerError(
                f"UnextractedConditionSpec.quantity_kind={self.quantity_kind!r} must be a "
                f"genuine QuantityKind member or None, not {type(self.quantity_kind).__name__}"
            )
        _require_int_occurrences(
            "UnextractedConditionSpec",
            label_occurrence=self.label_occurrence,
            statement_occurrence=self.statement_occurrence,
        )
        _reject_cell_with_occurrence(
            "UnextractedConditionSpec",
            label=(self.label_occurrence, self.label_cell),
            statement=(self.statement_occurrence, self.statement_cell),
        )


@dataclass(frozen=True, slots=True)
class DeviceClassSpec:
    """The apparatus the paper names, to be grounded as the condition set's subject."""

    label_quote: str
    label_occurrence: int | None = None

    def __post_init__(self) -> None:
        _require_int_occurrences("DeviceClassSpec", label_occurrence=self.label_occurrence)


@dataclass(frozen=True, slots=True)
class UnresolvedSubjectSpec:
    """A REFUSAL to name the apparatus, with the span that motivates the refusal.

    The refusal is still grounded: it points at the text that made the subject
    unresolvable, so a reader can check the refusal rather than take it on trust.
    """

    reason: SubjectRefusalReason
    reason_quote: str
    reason_occurrence: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, SubjectRefusalReason):
            raise ConditionSetProducerError(
                f"UnresolvedSubjectSpec.reason={self.reason!r} must be a genuine "
                f"SubjectRefusalReason member, not {type(self.reason).__name__}"
            )
        _require_int_occurrences("UnresolvedSubjectSpec", reason_occurrence=self.reason_occurrence)


def _cell_locator(cell: TableCellGrounding) -> TableCellLocator:
    """The ordinal-only :class:`TableCellLocator` for one validated cell grounding.

    Always carries a resolvable ``pdf_table_inventory_sha256`` -- the embedded
    inventory's own address -- never :class:`Absent`: an absent sha is a citation
    nothing can resolve, and a produced cell citation must be resolvable from the
    envelope's own bytes. The grounding this is built from is validated by
    :class:`_CellCiter` first, so building the locator here is pure assembly.
    """
    return TableCellLocator(
        table_key=cell.table_key,
        row=cell.row,
        col=cell.col,
        pdf_table_inventory_sha256=cell.inventory.inventory_sha256,
    )


class _CellCiter:
    """The producer's single authority on table-cell groundings for one envelope.

    Holds the target node and enforces, ACROSS ALL cells requested in one build:

    * the node is a ``PAPER_PDF`` -- only there does a cell have the fragment
      geometry an inventory describes, and only there may a locator carry a
      resolvable ``pdf_table_inventory_sha256``. Any other node kind would demand
      an ``Absent`` sha, which this producer never emits;
    * one cell is never asked to be two different strings (the ``TableCellLocator``
      has no sub-cell addressing, so a value and its unit sharing a cell is
      unrepresentable and must be refused, not laundered);
    * every cited cell EXISTS in its inventory, its inventory describes THIS
      document, and its whole text equals the whole grounded quote exactly.

    Validating the whole batch up front -- before any ref is built -- is what lets
    the one-cell-two-strings refusal win over the per-cell exact-equality message
    for the value-and-unit-share-a-cell case, which is the clearer diagnosis.
    """

    def __init__(self, node: SourceNode) -> None:
        self._node = node
        self._inventories: dict[str, EmbeddedTableInventory] = {}

    def validate(self, requests: tuple[tuple[str, str, TableCellGrounding], ...]) -> None:
        """Refuse any request that cannot become an honest citation; record the rest.

        ``requests`` is ``(owner, quote, cell)`` per grounded quote, where ``owner``
        names the claim and field for messages and ``quote`` is the exact string the
        cell must hold. Raises :class:`ConditionSetProducerError` on the first
        problem; on success every cited inventory is recorded for
        :meth:`table_inventories`.
        """
        if not requests:
            return
        if self._node.kind is not SourceNodeKind.PAPER_PDF:
            owners = sorted({owner for owner, _, _ in requests})
            raise ConditionSetProducerError(
                f"cell grounding requested by {owners} but the artifact's root node is "
                f"{self._node.kind.value!r}, not PAPER_PDF -- only a PDF node's cells have the fragment "
                "geometry a table inventory describes, and only there may a locator carry a resolvable "
                "pdf_table_inventory_sha256. Grounding a cell against any other node kind would require an "
                "Absent sha, which is a citation nothing can resolve and this producer never emits"
            )
        # A single cell cannot honestly be two different strings. Checked across the
        # whole batch first so the value-and-unit-share-a-cell spec is refused with
        # this precise message rather than a per-cell exact-equality complaint.
        by_cell: dict[tuple[str, int, int], set[str]] = {}
        for _owner, quote, cell in requests:
            by_cell.setdefault((cell.inventory.inventory_sha256, cell.row, cell.col), set()).add(quote)
        for (inv_sha, row, col), quotes in by_cell.items():
            if len(quotes) > 1:
                raise ConditionSetProducerError(
                    f"cell row={row}, col={col} of inventory {inv_sha!r} is grounded by two different strings "
                    f"{sorted(quotes)!r} -- a TableCellLocator has row, col, table_key and the inventory sha and "
                    "no sub-cell addressing whatsoever, so one cell cannot honestly be both; split the datum "
                    "across two cells or record it as a single grounded string"
                )
        for owner, quote, cell in requests:
            inventory = cell.inventory
            if inventory.raw_sha256 != self._node.sha256:
                raise ConditionSetProducerError(
                    f"{owner}: cell grounding cites inventory {inventory.inventory_sha256!r}, whose grid was "
                    f"derived from document {inventory.raw_sha256!r}, but the artifact's node is "
                    f"{self._node.sha256!r} -- the grid describes a different document than the one grounded here"
                )
            if not inventory.has_cell(row=cell.row, col=cell.col):
                raise ConditionSetProducerError(
                    f"{owner}: cell grounding names row={cell.row}, col={cell.col} in inventory "
                    f"{inventory.inventory_sha256!r}, whose grid has no such cell -- refusing to emit a citation "
                    "naming an ordinal the inventory never derived"
                )
            actual = inventory.cell_text(row=cell.row, col=cell.col)
            if actual is None:
                raise ConditionSetProducerError(
                    f"{owner}: cell row={cell.row}, col={cell.col} in inventory {inventory.inventory_sha256!r} "
                    "carries no readable text, so the exact-equality matching contract cannot be checked and the "
                    "citation cannot be shown true"
                )
            if actual != quote:
                raise ConditionSetProducerError(
                    f"{owner}: cell row={cell.row}, col={cell.col} in inventory {inventory.inventory_sha256!r} "
                    f"reads {actual!r}, but the grounded quote is {quote!r} -- the whole cell text must equal the "
                    "whole grounded string exactly (no substring, prefix or normalisation), or a value of '8' "
                    "could cite a cell reading '1-8'"
                )
            self._inventories[inventory.inventory_sha256] = inventory

    def table_inventories(self) -> tuple[EmbeddedTableInventory, ...]:
        """Every cited inventory, deduplicated by sha and sorted -- exactly what
        the envelope's T4 exact-cover and T5 sort-order validators require."""
        return tuple(sorted(self._inventories.values(), key=lambda inventory: inventory.inventory_sha256))


def _ref(
    text: str,
    quote: str,
    *,
    role: QuoteRole,
    occurrence: int | None,
    repairs: tuple[GlyphRepair, ...],
    cell: TableCellGrounding | None = None,
) -> SourceRef:
    """Ground ``quote`` and wrap the locator as a ``SourceRef``.

    When ``cell`` is given the quote is located at a table cell (already validated
    by :class:`_CellCiter`); otherwise it is searched in ``text`` as a character
    span, byte-for-byte as before -- the char-span path is unchanged.

    ``repairs`` is keyword-only and REQUIRED (no default): it is this producer's half
    of the lane seam -- the table lane's in-force glyph repairs for the document
    (``grounding.glyph_repairs``), passed through to :func:`ground_quote` so a citation
    into a character the text lane stores mis-decoded is refused, naming the mis-decode.
    Required rather than defaulted so a NEW char-span grounding cannot silently
    reintroduce a text lane that bypasses the repair path -- the omission is a type error
    at the call, not a quiet regression.
    """
    if cell is not None:
        return SourceRef(node_id=_ROOT_NODE_ID, locator=_cell_locator(cell))
    return SourceRef(
        node_id=_ROOT_NODE_ID,
        locator=ground_quote(text, quote, role=role, occurrence=occurrence, repairs=repairs),
    )


def _refuse_stitched(claim: GroundedScalarClaim, text: str) -> GroundedScalarClaim:
    """Refuse a scalar claim whose three grounds provably do not cohere.

    The producer holds the document text, so it is the earliest place this can
    be caught -- but it is deliberately NOT the only place. The same gate runs
    in :func:`~carmel.services.dataset_replay.replay_condition_set`, because a
    producer-side refusal says nothing about an envelope that was stored before
    this rule existed, or constructed by any route that does not come through
    here.

    Passing this gate is NOT a verification. It means one named refutation was
    attempted and did not fire; whether the paper actually predicates this label
    of this number remains unproven, and nothing downstream may upgrade it.
    """
    outcome = refute_stitched_claim(claim, text)
    if isinstance(outcome, StitchRefutation):
        raise ConditionSetProducerError(
            f"scalar claim {claim.claim_id!r} (label {claim.label_raw!r}, value "
            f"{claim.value.raw_text!r} {claim.value.unit_raw!r}) is refused: {outcome.reason}"
        )
    if isinstance(outcome, StitchGateUnrunnable):
        raise ConditionSetProducerError(
            f"scalar claim {claim.claim_id!r} cannot be checked for span stitching: "
            f"{outcome.reason}. This producer grounds every quote as a character span into "
            "one root node, so reaching this state means the claim was built by a route that "
            "does not hold that invariant -- it is refused rather than stored unchecked"
        )
    return claim


def _duplicate_ids(ids: list[str], *, owner: str) -> None:
    """Refuse duplicate ids across ALL of a condition set's collections.

    Two entries sharing an id makes every downstream per-claim finding ambiguous
    about which one it is about -- a replayer would report a path that names two
    different things. ``claim_id`` and ``statement_id`` share a single namespace;
    that is not this function's choice, it is what ``ConditionSetEnvelope``
    validates, and this refuses early so the caller is told which id collided
    rather than being handed a schema error from inside construction.
    """
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ConditionSetProducerError(
                f"duplicate {owner} {value!r}: every claim and statement in a condition set "
                "must have a unique id, or a per-claim finding cannot say which one it means"
            )
        seen.add(value)


def _cell_grounding_requests(
    scalars: tuple[ScalarConditionSpec, ...],
    categoricals: tuple[CategoricalConditionSpec, ...],
    unextracted: tuple[UnextractedConditionSpec, ...],
) -> tuple[tuple[str, str, TableCellGrounding], ...]:
    """Every ``(owner, quote, cell)`` a cell must be validated for, across all specs.

    ``owner`` names the claim and field for refusal messages; ``quote`` is the exact
    string the cell text must equal. Gathered up front so :class:`_CellCiter` can
    judge the whole batch at once -- collisions and exact-equality alike -- before a
    single ref is built.
    """
    requests: list[tuple[str, str, TableCellGrounding]] = []
    for scalar in scalars:
        if scalar.label_cell is not None:
            requests.append((f"scalar claim {scalar.claim_id!r} label", scalar.label_quote, scalar.label_cell))
        if scalar.value_cell is not None:
            requests.append((f"scalar claim {scalar.claim_id!r} value", scalar.value_quote, scalar.value_cell))
        if scalar.unit_cell is not None:
            requests.append((f"scalar claim {scalar.claim_id!r} unit", scalar.unit_quote, scalar.unit_cell))
    for categorical in categoricals:
        if categorical.label_cell is not None:
            requests.append(
                (f"categorical claim {categorical.claim_id!r} label", categorical.label_quote, categorical.label_cell)
            )
        if categorical.token_cell is not None:
            requests.append(
                (f"categorical claim {categorical.claim_id!r} token", categorical.token_quote, categorical.token_cell)
            )
    for statement in unextracted:
        if statement.label_cell is not None:
            requests.append(
                (f"unextracted statement {statement.statement_id!r} label", statement.label_quote, statement.label_cell)
            )
        if statement.statement_cell is not None:
            requests.append(
                (
                    f"unextracted statement {statement.statement_id!r} statement",
                    statement.statement_quote,
                    statement.statement_cell,
                )
            )
    return tuple(requests)


def _scalar_claim(
    spec: ScalarConditionSpec,
    text: str,
    *,
    document_source_context: SourceContext,
    document_glyph_health: GlyphHealth,
    document_glyph_repairs: tuple[GlyphRepair, ...],
) -> GroundedScalarClaim:
    """One :class:`GroundedScalarClaim`, cell- or char-grounded per its spec.

    The span-stitching refutation runs ONLY for a fully char-span claim. When any of
    the claim's three grounds is a table cell, the gate is structurally unrunnable --
    it reads a character window over one text span, and a cell has no such offsets --
    so ``refute_stitched_claim`` returns ``StitchGateUnrunnable`` and replay records
    the claim UNVERIFIABLE on that axis. Skipping it here recognises the gate's
    domain rather than relaxing it: the row/column adjacency a table encodes is not
    the text co-location the gate attacks, and forcing it to "refuse" would reject
    every honest table-grounded scalar. The cell citation was already validated by
    :class:`_CellCiter` (the cell exists, and its whole text equals this quote).
    """
    claim = GroundedScalarClaim(
        claim_id=spec.claim_id,
        label_raw=spec.label_quote,
        label_ref=_ref(
            text,
            spec.label_quote,
            role=QuoteRole.LABEL,
            occurrence=spec.label_occurrence,
            repairs=document_glyph_repairs,
            cell=spec.label_cell,
        ),
        value=_measured_value(
            text,
            spec,
            where=f"claim {spec.claim_id!r}",
            document_source_context=document_source_context,
            document_glyph_health=document_glyph_health,
            document_glyph_repairs=document_glyph_repairs,
            value_locator=_cell_locator(spec.value_cell) if spec.value_cell is not None else None,
            unit_locator=_cell_locator(spec.unit_cell) if spec.unit_cell is not None else None,
        ),
        # This producer reads no uncertainty from the document. That is a
        # NOT_EXTRACTED_YET refusal, not an assertion that the paper stated
        # none -- the two must never conflate.
        uncertainty=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
    )
    if spec.label_cell is None and spec.value_cell is None and spec.unit_cell is None:
        return _refuse_stitched(claim, text)
    return claim


def produce_condition_set_from_artifact(
    workspace_root: Path,
    *,
    sha256: str,
    attribution: ConditionAttribution,
    attribution_quote: str,
    subject: DeviceClassSpec | UnresolvedSubjectSpec,
    scalars: tuple[ScalarConditionSpec, ...] = (),
    categoricals: tuple[CategoricalConditionSpec, ...] = (),
    unextracted: tuple[UnextractedConditionSpec, ...] = (),
    attribution_occurrence: int | None = None,
) -> ConditionSetEnvelope:
    """Build one validated :class:`ConditionSetEnvelope` from a stored artifact.

    The vertical slice: authenticate ``raw.bin`` against the artifact's own
    sha256, select the ONE current extraction record, take the grounded text from
    it, ground every caller-stated quote in that text, and assemble an envelope
    that passes every schema validator. Construction runs pydantic's full
    validation -- nothing here uses ``model_construct``.

    The authentication preamble is NOT reimplemented here: it is
    :func:`~carmel.services.dataset_producer._prepare_grounding`, shared with the
    dataset producer, so that a fix to one producer's fail-closed path cannot
    silently miss the other's.

    Args:
        workspace_root: Workspace root holding the content-addressed store.
        sha256: The raw artifact's sha256.
        attribution: Whether these conditions are the paper's OWN experiment, a
            CITED third party's, or a SIMULATION. This is the caller's
            ASSERTION, recorded unverified -- ``attribution_quote`` grounds
            WHERE the assertion was read, never that it is correct.
        attribution_quote: The text the attribution was read from.
        subject: The apparatus, either named (:class:`DeviceClassSpec`) or
            explicitly refused (:class:`UnresolvedSubjectSpec`).
        scalars: Conditions resolving to one grounded number each.
        categoricals: Conditions resolving to one grounded token each.
        unextracted: Conditions REFUSED, each with its reason and its span.
        attribution_occurrence: Disambiguates a repeated ``attribution_quote``.

    Returns:
        A fully validated envelope.

    Raises:
        ConditionSetProducerError: Nothing was extracted at all, ids collide, a
            spec field is the wrong type, or a value is not a bare numeral / its
            unit is unknown for its quantity kind.
        DatasetProducerError: The artifact is missing, legacy, corrupt, lossily
            extracted, or has no usable current extraction record.
        QuoteGroundingError: A quote is absent from the document, or occurs more
            than once and was not disambiguated.
    """
    if not scalars and not categoricals and not unextracted:
        # An envelope asserting no condition at all is not a modest result, it is
        # a claim that the paper stated no conditions -- which this producer has
        # no way to establish. Refuse rather than emit a vacuously VERIFIED
        # envelope, which is precisely the overclaim shape this system exists to
        # prevent.
        raise ConditionSetProducerError(
            f"artifact {sha256!r}: refusing to produce a condition set with no scalar claims, "
            "no categorical claims and no recorded refusals -- an empty condition set asserts "
            "that the paper stated no conditions, which grounding cannot establish"
        )
    # ONE namespace across all three collections, because that is what
    # ConditionSetEnvelope itself enforces. Checking the two kinds separately
    # let a claim_id collide with a statement_id and surface as a pydantic
    # ValidationError from inside construction -- a late, badly-located refusal
    # for a caller error this producer can name precisely.
    _duplicate_ids(
        [s.claim_id for s in scalars] + [c.claim_id for c in categoricals] + [u.statement_id for u in unextracted],
        owner="id",
    )
    if not isinstance(attribution, ConditionAttribution):
        raise ConditionSetProducerError(
            f"attribution={attribution!r} must be a genuine ConditionAttribution member, not "
            f"{type(attribution).__name__} -- ConditionAttribution is a StrEnum, so a plain "
            "string equal to a member's value would compare `==` equal without being that member"
        )
    _require_int_occurrences("produce_condition_set_from_artifact", attribution_occurrence=attribution_occurrence)

    grounding = _prepare_grounding(
        workspace_root, sha256, envelope_noun="condition set", envelope_subject="A condition set"
    )
    text = grounding.text
    # The lane seam, on the condition-set side: every char-span grounding below is handed the
    # table lane's in-force glyph repairs for this document, so a citation into a character the
    # text lane stores mis-decoded is refused with the mis-decode named (see ground_quote).
    repairs = grounding.glyph_repairs

    # The single authority on cell citations. Every cell grounding requested across
    # every spec is validated as ONE batch here -- before any ref is built -- so a
    # cell that does not exist, whose text differs from its quote, or that a value and
    # a unit both claim, is refused with the clearest message, and the inventories the
    # envelope must embed are collected exactly once. The subject and attribution are
    # deliberately NOT cell-groundable: they are the set's provenance frame, not a
    # datum read out of a grid.
    citer = _CellCiter(grounding.graph.node(_ROOT_NODE_ID))
    citer.validate(_cell_grounding_requests(scalars, categoricals, unextracted))

    resolved_subject: DeviceClassDeclaration | UnresolvedSubject
    if isinstance(subject, DeviceClassSpec):
        resolved_subject = DeviceClassDeclaration(
            label_raw=subject.label_quote,
            label_ref=_ref(
                text,
                subject.label_quote,
                role=QuoteRole.LABEL,
                occurrence=subject.label_occurrence,
                repairs=repairs,
            ),
        )
    else:
        resolved_subject = UnresolvedSubject(
            reason=subject.reason,
            reason_ref=_ref(
                text,
                subject.reason_quote,
                role=QuoteRole.LABEL,
                occurrence=subject.reason_occurrence,
                repairs=repairs,
            ),
        )

    scalar_claims = tuple(
        _scalar_claim(
            spec,
            text,
            document_source_context=grounding.document_source_context,
            document_glyph_health=grounding.document_glyph_health,
            document_glyph_repairs=repairs,
        )
        for spec in scalars
    )
    categorical_claims = tuple(
        GroundedCategoricalClaim(
            claim_id=spec.claim_id,
            label_raw=spec.label_quote,
            label_ref=_ref(
                text,
                spec.label_quote,
                role=QuoteRole.LABEL,
                occurrence=spec.label_occurrence,
                repairs=repairs,
                cell=spec.label_cell,
            ),
            token_raw=spec.token_quote,
            token_ref=_ref(
                text,
                spec.token_quote,
                role=QuoteRole.VALUE,
                occurrence=spec.token_occurrence,
                repairs=repairs,
                cell=spec.token_cell,
            ),
        )
        for spec in categoricals
    )
    unextracted_statements = tuple(
        UnextractedConditionStatement(
            statement_id=spec.statement_id,
            label_raw=spec.label_quote,
            label_ref=_ref(
                text,
                spec.label_quote,
                role=QuoteRole.LABEL,
                occurrence=spec.label_occurrence,
                repairs=repairs,
                cell=spec.label_cell,
            ),
            statement_ref=_ref(
                text,
                spec.statement_quote,
                role=QuoteRole.VALUE,
                occurrence=spec.statement_occurrence,
                repairs=repairs,
                cell=spec.statement_cell,
            ),
            reason=spec.reason,
            quantity_kind=(
                spec.quantity_kind if spec.quantity_kind is not None else Absent(reason=AbsenceReason.NOT_EXTRACTED_YET)
            ),
        )
        for spec in unextracted
    )

    return ConditionSetEnvelope(
        source_graph=grounding.graph,
        # Only a MeasuredValue cites a conversion table, so a condition set that
        # resolved no scalar claims must embed NONE: the schema refuses a
        # decorative table as "unearned provenance", and it is right to. A
        # refusal-only condition set is a legitimate result, and it may not
        # carry provenance for a conversion it never performed.
        conversion_tables=(_ACTIVE.embedded,) if scalar_claims else (),
        # Exactly the inventories this build's TABLE_CELL locators cite -- collected,
        # deduplicated and sorted by _CellCiter, filled from the specs and NEVER from a
        # store lookup at replay time. Empty when nothing was cell-grounded, which keeps
        # a pure char-span condition set byte-identical to before: T4 refuses a decorative
        # inventory nothing cites for the same reason conversion_tables above refuses a
        # decorative table.
        table_inventories=citer.table_inventories(),
        subject=resolved_subject,
        attribution=attribution,
        attribution_ref=_ref(
            text, attribution_quote, role=QuoteRole.LABEL, occurrence=attribution_occurrence, repairs=repairs
        ),
        scalar_claims=scalar_claims,
        categorical_claims=categorical_claims,
        unextracted=unextracted_statements,
    )
