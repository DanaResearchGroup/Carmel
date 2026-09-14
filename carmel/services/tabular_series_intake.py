"""Carry a table already classified MEASURED, through an injected agent, to a
stored :class:`~carmel.schemas.datasets.DatasetEnvelope`.

This is the lane-agnostic sibling of
:func:`carmel.services.proposal_intake.tabular_series_from_proposal`: that carrier
turns an already-validated :class:`~carmel.agents.extraction_agent.TabularSeriesProposal`
into a produced envelope; this module is the step ABOVE it that drives the agent,
gates on the table's classification, and owns storage. It imports nothing from
``pdf_*`` and knows nothing of a page number, a footprint, or a lane's payload key --
every caller-specific fact (which document, which grid, which classification) arrives
already built, so the same function serves the PDF driver
(:mod:`carmel.services.general_table_report`) and any future lane with an
``EmbeddedTableInventory`` and a :class:`~carmel.services.table_data_discriminator.TableClassification`
in hand.

DESIGN DECISION recorded here, as the brief that commissioned this module requires:
both the structural refusal (step 3 below) and the stage/verify/promote storage
discipline (step 5) live IN :func:`series_from_classified_grid`, not composed by the
caller. The caller (the PDF driver, or any future lane) already assembles the
inputs this function needs from data it alone can produce (page, footprint, staged
bytes); asking it to also re-implement "read the produced envelope's own citations
and refuse on ``unmeasured_columns``" or "never store a value that fails to replay"
would duplicate a check across every lane driver, with no lane-specific reason either
half needs to vary. Keeping both here means there is exactly one place that can get
either wrong.

THE ONE LEGITIMATE STRUCTURAL REFUSAL, and why it is INCOMPLETE. Step 3 refuses a
proposal whose OBSERVATION axis resolved into a column the classifier separately
marked :attr:`~carmel.services.table_data_discriminator.TableClassification.unmeasured_columns`
-- e.g. a reaction-key or formula column the classifier already has independent,
column-scoped evidence is NOT measured data. This check is SOUND: every column it
refuses genuinely carries a ``NOT_MEASURED``-polarity ground written by the
classifier's own structural read of the document, so it can never invent a refusal.
It is NOT COMPLETE: a value column with no positive OR negative structural signal
(the ``UNDECIDED``-flavoured middle the classifier itself names) is not caught here,
because :attr:`TableClassification.unmeasured_columns` is silent about it -- silence
is not evidence either way. This check deliberately never consults
:attr:`~carmel.services.table_data_discriminator.TableClassification.measured_columns`
(gating an OBSERVATION axis on the ABSENCE of a positive signal is the inversion bug
this module exists to avoid: most legitimately measured value columns carry no
column-scoped positive ground of their own, only the table-level ones the caption or
a coordinate column's sweep supplies) and it never gates on a quantity-name
whitelist (the agent's ``quantity_kind`` is a semantic assertion this module records,
never verifies).
"""

from __future__ import annotations

from pathlib import Path

from carmel.agents.bridge import CarmelAgent
from carmel.agents.extraction_agent import TabularSeriesProposal
from carmel.schemas.datasets import (
    AxisRole,
    CaptionLabelKey,
    DatasetEnvelope,
    EmbeddedTableInventory,
    MeasuredValue,
    TableCellLocator,
)
from carmel.services.dataset_bridge import store_dataset_envelope
from carmel.services.dataset_replay import replay_envelope
from carmel.services.dataset_store import StoredDataset
from carmel.services.literature import estimated_tokens_for
from carmel.services.proposal_intake import build_tabular_series_prompt, tabular_series_from_proposal
from carmel.services.table_data_discriminator import DataVerdict, TableClassification

__all__ = [
    "TabularSeriesIntakeError",
    "series_from_classified_grid",
]


class TabularSeriesIntakeError(Exception):
    """A typed, closed refusal from :func:`series_from_classified_grid`.

    Every refusal this module raises directly (as opposed to one it propagates
    unchanged from the carrier, the resolver, the producer or the agent bridge)
    is this type, and names the step that refused, the document, and the
    candidate table."""


def _render_grid_text(inventory: EmbeddedTableInventory) -> str:
    """Render ``inventory``'s cells as plain, position-labelled lines.

    One line per non-empty cell, ``row {r} col {c}: {text}``, in the same
    ``(row, col)`` order :meth:`EmbeddedTableInventory.grid_cells` returns them.
    A cell with no genuine text (``grid_cells`` reporting ``None``) is omitted
    rather than rendered as an empty quote, so the model is never invited to
    quote a header this grid does not actually carry.

    The cell text is rendered VERBATIM -- never ``repr()``/``{!r}``, which would
    wrap it in quotes and backslash-escape any backslash, tab or newline. The
    resolver matches ``header_quote`` whole-cell against this exact cell text, and
    the persona promises the model the grid is printed "character for character";
    a ``repr``-escaped rendering would break that promise for any header carrying a
    quote or a backslash -- a model echoing what it was shown would quote text the
    cell does not contain and be refused. Everything up to and including the first
    ``": "`` is renderer punctuation; everything after it, to end of line, is the
    cell's exact bytes. (Table-cell text in these lanes is single-line, so a newline
    inside a cell -- the one case this line-oriented framing could not represent --
    does not arise.)
    """
    lines = [f"row {row} col {col}: {text}" for row, col, text in inventory.grid_cells() if text is not None]
    return "\n".join(lines)


def _cited_column(envelope: DatasetEnvelope, axis_id: str) -> int | None:
    """The column a resolved OBSERVATION axis's cited cells live in.

    Reads the AUTHORITATIVE cell locator the producer actually wrote onto every
    point's observation for ``axis_id`` -- never re-derived from the grid and
    never taken from any model-asserted ordinal. Returns ``None`` when the axis
    has no points citing a :class:`TableCellLocator` (an absent observation, or
    a value grounded some other way), in which case there is nothing for the
    unmeasured-column gate below to check.

    Raises:
        TabularSeriesIntakeError: two points cite the SAME axis at two
            different columns -- a shape the producer should never emit, so
            this is a fail-closed sanity check rather than an expected case.
    """
    columns: set[int] = set()
    for series in envelope.series:
        for point in series.points:
            for observation in point.observations:
                if observation.axis_id != axis_id:
                    continue
                value = observation.value
                if not isinstance(value, MeasuredValue):
                    continue
                locator = value.value_ref.locator
                if isinstance(locator, TableCellLocator):
                    columns.add(locator.col)
    if not columns:
        return None
    if len(columns) > 1:
        raise TabularSeriesIntakeError(
            f"axis {axis_id!r} cites more than one column across its points ({sorted(columns)!r}); "
            "the unmeasured-column gate expects one axis to resolve to one column"
        )
    return next(iter(columns))


def series_from_classified_grid(
    workspace_root: Path,
    *,
    agent: CarmelAgent,
    expected_sha256: str,
    table_key: CaptionLabelKey,
    inventory: EmbeddedTableInventory,
    classification: TableClassification,
    document_text: str,
    estimated_tokens: int | None = None,
) -> StoredDataset:
    """Drive ``agent`` over one classified table and store the resulting series.

    Five steps, each a typed, closed-failure refusal naming the step, the
    document and the candidate table:

    1. Refuse unless ``classification.verdict`` is
       :attr:`~carmel.services.table_data_discriminator.DataVerdict.MEASURED`.
    2. Build the prompt (:func:`~carmel.services.proposal_intake.build_tabular_series_prompt`,
       its grid text rendered from ``inventory.grid_cells()``) and run ``agent``.
       A schema-invalid response surfaces as :class:`~carmel.agents.bridge.AgentBridgeError`,
       propagated unchanged.
    3. THE ONE LEGITIMATE STRUCTURAL REFUSAL -- see the module docstring. After
       the proposal is carried to a produced envelope, for every OBSERVATION
       axis, read its AUTHORITATIVE cited column off the produced envelope
       (never re-derived, never a model ordinal); refuse if that column is in
       ``classification.unmeasured_columns``. Sound but INCOMPLETE: silence
       (no ground either way) is not caught here, and NEVER
       ``classification.measured_columns``.
    4. Carry the proposal through the unchanged
       :func:`~carmel.services.proposal_intake.tabular_series_from_proposal` --
       already done to reach step 3's envelope, so this step is folded into
       step 2/3's flow rather than repeated.
    5. Never store a non-replaying value: replay the produced envelope, in
       memory, against the evidence already on disk at ``workspace_root``
       (:func:`~carmel.services.dataset_replay.replay_envelope`) BEFORE it is
       ever handed to the store. A clean replay has
       ``report.evidence_failures == ()``; anything else is a typed refusal
       and NOTHING is stored. Only a clean replay reaches
       :func:`~carmel.services.dataset_bridge.store_dataset_envelope`, whose
       own contract additionally refuses (before writing) a payload that
       would not itself read back byte-for-byte -- the "stage, verify, promote"
       discipline :func:`carmel.services.general_table_report._classify_and_maybe_store`
       uses for inventory records, mirrored here for a dataset envelope.

    Args:
        workspace_root: Workspace root holding the content-addressed store and
            the document's evidence (``raw.bin`` and current extraction).
        agent: A :class:`~carmel.agents.bridge.CarmelAgent` whose
            ``output_schema`` is :class:`~carmel.agents.extraction_agent.TabularSeriesProposal`.
            Drive it with :class:`carmel.agents.models.MockModel` in tests --
            never a live model.
        expected_sha256: The sha256 of the document the caller prompted with;
            the authority ``proposal.artifact_sha256`` must equal (checked by
            the carrier).
        table_key: The caller-supplied identity of the table whose grid is
            ``inventory``; the authority ``proposal.table_label`` must equal
            (checked by the carrier).
        inventory: The caller-supplied grid for the candidate table.
        classification: The table's :class:`~carmel.services.table_data_discriminator.TableClassification`,
            already computed by the caller.
        document_text: The document's current extraction text, exactly as
            :func:`~carmel.services.proposal_intake.current_extraction_text`
            returns it -- the same bytes the produced envelope's char-span
            grounds (unit quotes) will be checked against.
        estimated_tokens: Budget reservation for the agent call. Defaults to
            :func:`carmel.services.literature.estimated_tokens_for` applied to
            the built prompt, sized to the grid and document text actually sent.

    Returns:
        The stored dataset's address
        (:class:`~carmel.services.dataset_store.StoredDataset`), AFTER its envelope has
        replayed cleanly. This function owns the single store; the caller reuses this
        address rather than storing the envelope a second time (a redundant store would
        be a second, unprotected write). Callers that need the envelope itself load it
        back with :func:`~carmel.services.dataset_bridge.load_dataset_envelope`.

    Raises:
        TabularSeriesIntakeError: The table is not MEASURED, an OBSERVATION axis
            resolved into an unmeasured column, or the produced envelope failed
            to replay cleanly.
        AgentBridgeError: The agent's response did not validate against
            ``TabularSeriesProposal``.
        ProposalIntakeError / TabularSeriesResolutionError /
        TabularDatasetProducerError / DatasetProducerError / QuoteGroundingError:
            Propagated unchanged from the carrier, resolver or producer -- see
            :func:`~carmel.services.proposal_intake.tabular_series_from_proposal`.
    """
    label = f"table {table_key.label!r} of document {expected_sha256!r}"

    if classification.verdict is not DataVerdict.MEASURED:
        raise TabularSeriesIntakeError(
            f"series_from_classified_grid: refusing {label} -- classification verdict is "
            f"{classification.verdict.value!r}, not measured. Only a MEASURED table may be "
            "handed to the extraction agent for a series read"
        )

    grid_text = _render_grid_text(inventory)
    prompt = build_tabular_series_prompt(
        artifact_sha256=expected_sha256,
        table_label=table_key.label,
        grid_text=grid_text,
        document_text=document_text,
    )
    reservation_tokens = estimated_tokens if estimated_tokens is not None else estimated_tokens_for(prompt)
    result = agent.run(prompt, estimated_tokens=reservation_tokens)
    proposal = TabularSeriesProposal.model_validate(result.output)

    envelope = tabular_series_from_proposal(
        workspace_root,
        proposal,
        expected_sha256=expected_sha256,
        table_key=table_key,
        inventory=inventory,
    )

    unmeasured = set(classification.unmeasured_columns)
    for axis in proposal.axes:
        if axis.role is not AxisRole.OBSERVATION:
            continue
        column = _cited_column(envelope, axis.axis_id)
        if column is not None and column in unmeasured:
            raise TabularSeriesIntakeError(
                f"series_from_classified_grid: refusing {label} -- observation axis "
                f"{axis.axis_id!r} (header {axis.header_quote!r}) resolved into column {column}, "
                "which the table classifier separately marked unmeasured "
                f"(unmeasured_columns={classification.unmeasured_columns!r}). This check is SOUND "
                "(the column carries a real NOT_MEASURED-polarity ground) but INCOMPLETE (silence "
                "elsewhere is not evidence either way); it never gates on measured_columns"
            )

    report = replay_envelope(workspace_root, envelope)
    if report.evidence_failures != ():
        raise TabularSeriesIntakeError(
            f"series_from_classified_grid: refusing {label} -- produced series does not replay "
            f"cleanly against the evidence store ({len(report.evidence_failures)} evidence "
            "failure(s)); nothing was stored. First failure: "
            f"{report.evidence_failures[0].reason!r}"
        )

    return store_dataset_envelope(workspace_root, envelope)
