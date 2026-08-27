"""The condition-set producer emitting TABLE_CELL locators (issue I-022).

The schema could already express a claim grounded in a table cell; nothing could
PRODUCE one. These tests drive the producer -- never a hand-assembled envelope --
and assert the properties the ticket's verifier names: a produced value can cite a
cell, all three spec kinds can, every produced cell citation carries a resolvable
inventory sha, a cell the grid lacks is REFUSED, one cell cannot be two strings,
and a produced artifact survives produce -> store -> load -> replay.

Every fixture is SYNTHETIC. The inventories are hand-built (a real grid needs the
pypdf-dependent fragment lane, and the schema validates a cell citation's SHAPE,
never that the grid is real -- see ``tests.table_inventory_fixtures``); the only
thing pinned to reality is the exact-equality contract, checked here directly
because on this base branch replay cannot yet check it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carmel.schemas.datasets import (
    CaptionLabelKey,
    ConditionAttribution,
    LocatorKind,
    TableCellLocator,
    UnextractedReason,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.condition_set_bridge import (
    load_condition_set_envelope,
    store_condition_set_envelope,
)
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    ConditionSetProducerError,
    DeviceClassSpec,
    ScalarConditionSpec,
    TableCellGrounding,
    UnextractedConditionSpec,
    produce_condition_set_from_artifact,
)
from carmel.services.dataset_replay import SemanticGap, replay_condition_set
from tests.table_inventory_fixtures import make_embedded_inventory_with_texts
from tests.test_dataset_producer import _store_synthetic_artifact

#: Running prose carrying ONLY the quotes that stay char-span grounded: the
#: apparatus (the set's subject) and the attribution. Every datum below comes from
#: the table, so its text does not appear here -- which is exactly the point of a
#: cell citation.
_METHODS_TEXT = (
    "2. Experimental methods\n"
    "Measurements were carried out in a jet-stirred reactor of fused silica.\n"
    "Table 1 lists the measurement conditions used across the campaign.\n"
)

_TABLE_KEY = CaptionLabelKey(label="Table 1")


def _table(raw_sha256: str) -> object:
    """A synthetic Table-1 inventory whose cells hold the exact quotes below.

    A 3-column grid: col 0 the printed label, col 1 the value, col 2 the unit,
    one row per datum. The equivalence-ratio row's value is a RANGE, so it is a
    cell whose whole text is the range string -- the shape the unextracted lane
    grounds.
    """
    return make_embedded_inventory_with_texts(
        raw_sha256=raw_sha256,
        cell_texts={
            (0, 0): "initial temperature",
            (0, 1): "823",
            (0, 2): "K",
            (1, 0): "diluent",
            (1, 1): "CO2",
            (2, 0): "equivalence ratio",
            (2, 1): "0.6 to 1.4",
        },
    )


def _full_coverage(tmp_path: Path) -> tuple[object, object]:
    """Produce an envelope exercising a cell citation in ALL THREE spec kinds.

    Returns ``(envelope, inventory)``. The scalar grounds its label, value and unit
    in cells; the categorical grounds its token in a cell; the unextracted grounds
    BOTH its label and its statement in cells.
    """
    stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
    inventory = _table(stored.sha256)
    envelope = produce_condition_set_from_artifact(
        tmp_path,
        sha256=stored.sha256,
        attribution=ConditionAttribution.OWN_EXPERIMENT,
        attribution_quote="Measurements were carried out",
        subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
        scalars=(
            ScalarConditionSpec(
                claim_id="initial_temperature",
                label_quote="initial temperature",
                quantity_kind=units.QuantityKind.TEMPERATURE,
                value_quote="823",
                unit_quote="K",
                label_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=0, inventory=inventory),
                value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                unit_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=2, inventory=inventory),
            ),
        ),
        categoricals=(
            CategoricalConditionSpec(
                claim_id="diluent",
                label_quote="diluent",
                token_quote="CO2",
                label_cell=TableCellGrounding(table_key=_TABLE_KEY, row=1, col=0, inventory=inventory),
                token_cell=TableCellGrounding(table_key=_TABLE_KEY, row=1, col=1, inventory=inventory),
            ),
        ),
        unextracted=(
            UnextractedConditionSpec(
                statement_id="equivalence_ratio",
                label_quote="equivalence ratio",
                statement_quote="0.6 to 1.4",
                reason=UnextractedReason.VALUE_RANGE,
                quantity_kind=units.QuantityKind.EQUIVALENCE_RATIO,
                label_cell=TableCellGrounding(table_key=_TABLE_KEY, row=2, col=0, inventory=inventory),
                statement_cell=TableCellGrounding(table_key=_TABLE_KEY, row=2, col=1, inventory=inventory),
            ),
        ),
    )
    return envelope, inventory


def _cell_refs(envelope: object) -> list[tuple[str, TableCellLocator]]:
    return [
        (path, ref.locator) for path, ref in iter_source_refs(envelope) if isinstance(ref.locator, TableCellLocator)
    ]


class TestAProducedValueCitesACell:
    """Verifier 1: a PRODUCED artifact carries a value whose ref is a TABLE_CELL
    locator citing an embedded inventory, and validates."""

    def test_a_produced_scalar_value_ref_is_a_table_cell(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        inventory = _table(stored.sha256)

        envelope = produce_condition_set_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            attribution=ConditionAttribution.OWN_EXPERIMENT,
            attribution_quote="Measurements were carried out",
            subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
            scalars=(
                ScalarConditionSpec(
                    claim_id="initial_temperature",
                    label_quote="initial temperature",
                    quantity_kind=units.QuantityKind.TEMPERATURE,
                    value_quote="823",
                    unit_quote="K",
                    label_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=0, inventory=inventory),
                    value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                    unit_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=2, inventory=inventory),
                ),
            ),
        )

        # Construction ran pydantic's full validation (no model_construct), so the
        # envelope validating IS the assertion -- V8/T4/T5 all passed on producer output.
        value_locator = envelope.scalar_claims[0].value.value_ref.locator
        assert isinstance(value_locator, TableCellLocator)
        assert value_locator.kind is LocatorKind.TABLE_CELL
        assert value_locator.row == 0
        assert value_locator.col == 1
        # The embedded inventory covers exactly the one cited grid.
        assert len(envelope.table_inventories) == 1
        assert envelope.table_inventories[0].inventory_sha256 == value_locator.pdf_table_inventory_sha256

    def test_the_scalars_char_span_path_is_unchanged_when_no_cell_is_given(self, tmp_path: Path) -> None:
        """The added path does not disturb the old one: a value with no cell is still
        a char span, and nothing is embedded."""
        text = _METHODS_TEXT + "The initial temperature was 823 K in every run.\n"
        stored = _store_synthetic_artifact(tmp_path, text)

        envelope = produce_condition_set_from_artifact(
            tmp_path,
            sha256=stored.sha256,
            attribution=ConditionAttribution.OWN_EXPERIMENT,
            attribution_quote="Measurements were carried out",
            subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
            scalars=(
                ScalarConditionSpec(
                    claim_id="initial_temperature",
                    label_quote="initial temperature",
                    quantity_kind=units.QuantityKind.TEMPERATURE,
                    value_quote="823",
                    unit_quote="K",
                ),
            ),
        )

        assert envelope.scalar_claims[0].value.value_ref.locator.kind is LocatorKind.CHAR_SPAN
        assert envelope.table_inventories == ()


class TestAllThreeSpecKindsCanCiteACell:
    """Verifier 2: scalar, categorical, and unextracted (BOTH refs) can cell-ground."""

    def test_every_spec_kind_grounds_against_a_cell(self, tmp_path: Path) -> None:
        envelope, _ = _full_coverage(tmp_path)

        scalar = envelope.scalar_claims[0]
        assert scalar.label_ref.locator.kind is LocatorKind.TABLE_CELL
        assert scalar.value.value_ref.locator.kind is LocatorKind.TABLE_CELL
        assert scalar.value.unit_ref.locator.kind is LocatorKind.TABLE_CELL

        categorical = envelope.categorical_claims[0]
        assert categorical.label_ref.locator.kind is LocatorKind.TABLE_CELL
        assert categorical.token_ref.locator.kind is LocatorKind.TABLE_CELL

        statement = envelope.unextracted[0]
        assert statement.label_ref.locator.kind is LocatorKind.TABLE_CELL
        assert statement.statement_ref.locator.kind is LocatorKind.TABLE_CELL

    def test_the_unextracted_statement_grounds_both_of_its_refs(self, tmp_path: Path) -> None:
        """The clause most likely to be missed: an unextracted VALUE_RANGE keeps a
        cell citation on both label_ref and statement_ref, so a refused range still
        points at the cells it refused."""
        envelope, _ = _full_coverage(tmp_path)

        statement = envelope.unextracted[0]
        assert statement.reason is UnextractedReason.VALUE_RANGE
        for locator in (statement.label_ref.locator, statement.statement_ref.locator):
            assert isinstance(locator, TableCellLocator)
            assert isinstance(locator.pdf_table_inventory_sha256, str)


class TestEveryProducedCitationIsResolvable:
    """Verifier 3: a produced table-cell citation never has an Absent inventory sha."""

    def test_no_produced_cell_locator_has_an_absent_sha(self, tmp_path: Path) -> None:
        envelope, inventory = _full_coverage(tmp_path)

        cell_refs = _cell_refs(envelope)
        assert len(cell_refs) == 7  # 3 scalar + 2 categorical + 2 unextracted
        for path, locator in cell_refs:
            sha = locator.pdf_table_inventory_sha256
            assert isinstance(sha, str), f"{path} carries a non-str (Absent?) inventory sha: {sha!r}"
            assert len(sha) == 64
            assert sha == inventory.inventory_sha256


class TestACellTheGridLacksIsRefused:
    """Verifier 4: a PRODUCED citation naming a (row, col) the inventory does not
    contain is refused, naming the missing ordinal -- driven THROUGH the producer,
    not by hand-constructing a bad locator (which the validator already refuses)."""

    def test_producing_a_missing_cell_is_refused_and_names_the_ordinal(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        inventory = _table(stored.sha256)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
                scalars=(
                    ScalarConditionSpec(
                        claim_id="initial_temperature",
                        label_quote="initial temperature",
                        quantity_kind=units.QuantityKind.TEMPERATURE,
                        value_quote="823",
                        unit_quote="K",
                        # Row 9 / col 7 is not in the grid the fixture built.
                        value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=9, col=7, inventory=inventory),
                        unit_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=2, inventory=inventory),
                    ),
                ),
            )

        message = str(excinfo.value)
        assert "row=9" in message and "col=7" in message
        assert "no such cell" in message


class TestTheExactEqualityContract:
    """The settled matching contract, checked at the producer because replay cannot
    on this base branch: whole cell text must equal whole grounded string."""

    def test_a_quote_that_is_a_substring_of_the_cell_is_refused(self, tmp_path: Path) -> None:
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        # The cell reads "1-8"; a value of "8" must NOT be allowed to cite it.
        inventory = make_embedded_inventory_with_texts(
            raw_sha256=stored.sha256, cell_texts={(0, 0): "pressure", (0, 1): "1-8"}
        )

        with pytest.raises(ConditionSetProducerError) as excinfo:
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
                scalars=(
                    ScalarConditionSpec(
                        claim_id="pressure",
                        label_quote="pressure",
                        quantity_kind=units.QuantityKind.PRESSURE,
                        value_quote="8",
                        unit_quote="atm",
                        value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                    ),
                ),
            )

        message = str(excinfo.value)
        assert "'1-8'" in message
        assert "'8'" in message
        assert "exactly" in message

    def test_a_cell_cannot_be_two_different_strings(self, tmp_path: Path) -> None:
        """Consequence 2: value and unit sharing one cell is unrepresentable -- a
        TableCellLocator has no sub-cell addressing -- so the producer refuses,
        naming the cell and both strings."""
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT)
        inventory = make_embedded_inventory_with_texts(
            raw_sha256=stored.sha256, cell_texts={(0, 0): "temperature", (0, 1): "298 K"}
        )

        with pytest.raises(ConditionSetProducerError) as excinfo:
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
                scalars=(
                    ScalarConditionSpec(
                        claim_id="temperature",
                        label_quote="temperature",
                        quantity_kind=units.QuantityKind.TEMPERATURE,
                        value_quote="298",
                        unit_quote="K",
                        # Both point at the SAME cell (0, 1) reading "298 K".
                        value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                        unit_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                    ),
                ),
            )

        message = str(excinfo.value)
        assert "row=0" in message and "col=1" in message
        assert "'298'" in message and "'K'" in message


class TestOccurrenceAndCellAreMutuallyExclusive:
    def test_supplying_both_an_occurrence_and_a_cell_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ConditionSetProducerError) as excinfo:
            ScalarConditionSpec(
                claim_id="t",
                label_quote="initial temperature",
                quantity_kind=units.QuantityKind.TEMPERATURE,
                value_quote="823",
                unit_quote="K",
                value_occurrence=0,
                value_cell=TableCellGrounding(
                    table_key=_TABLE_KEY,
                    row=0,
                    col=1,
                    inventory=make_embedded_inventory_with_texts(raw_sha256="a" * 64, cell_texts={(0, 1): "823"}),
                ),
            )
        assert "cannot be grounded both ways" in str(excinfo.value)


class TestProduceStoreLoadReplay:
    """Verifier 5: produce -> store -> load -> replay, reporting BOTH outcomes.

    The "separately scheduled work" this test's docstring once anticipated -- a
    replayer that checks cell locators -- has landed (PR #24's cell-text gate,
    then I-028 pairing ``statement_ref`` with ``statement_raw``). So the produced
    cell refs are now COMPARED against the embedded grid rather than filed as
    LOCATION_UNRESOLVED. The statement in particular carries its own words now, so
    it is compared like every label/token, not left as an unchecked meaning.
    """

    def test_the_round_trip_runs_and_the_cell_refs_are_compared(self, tmp_path: Path) -> None:
        envelope, _ = _full_coverage(tmp_path)

        stored = store_condition_set_envelope(tmp_path, envelope)
        loaded = load_condition_set_envelope(tmp_path, stored.sha256)
        assert loaded == envelope  # the embedded inventory survives the round trip verbatim

        report = replay_condition_set(tmp_path, loaded)

        # Every cited cell (scalar label/value/unit, categorical label/token, the
        # unextracted statement's label AND its now-paired statement) is compared
        # whole-cell against the embedded grid and matches -- 7 cells in all.
        assert report.checked_table_cells == 7
        # And NONE of them is filed as an unchecked semantic claim: the statement's
        # own path in particular is gone from that axis, because its words are
        # recorded and were compared, not merely located.
        assert all(claim.claim_path != "unextracted[0]" for claim in report.unchecked_semantic_claims)
        location_unresolved = [
            claim for claim in report.unchecked_semantic_claims if claim.gap is SemanticGap.LOCATION_UNRESOLVED
        ]
        assert location_unresolved == []
        # Both outcomes are read and reported, never asserted green. evidence_outcome
        # stays honest about what the SYNTHETIC bytes cannot do -- reproduce the
        # embedded inventory grid -- which is a separate check from the cell-text
        # comparison and keeps the overall verdict short of VERIFIED.
        assert report.evidence_outcome is not None
        assert report.overall_outcome is not None


class TestCellGroundingIsRefusedAgainstANonPdfNode:
    """The ``PAPER_PDF`` restriction at ``_CellCiter.validate()``, exercised by
    observing it FIRE -- not merely that something raised. It was unreached by any
    test before, which is why a reviewer had to find the docstring's contradiction
    of it by reading rather than by running."""

    def test_grounding_a_cell_against_a_jats_node_names_the_wrong_kind(self, tmp_path: Path) -> None:
        # An application/xml artifact yields a JATS_XML root node, not PAPER_PDF.
        stored = _store_synthetic_artifact(tmp_path, _METHODS_TEXT, content_type="application/xml", extractor="xml")
        inventory = _table(stored.sha256)

        with pytest.raises(ConditionSetProducerError) as excinfo:
            produce_condition_set_from_artifact(
                tmp_path,
                sha256=stored.sha256,
                attribution=ConditionAttribution.OWN_EXPERIMENT,
                attribution_quote="Measurements were carried out",
                subject=DeviceClassSpec(label_quote="jet-stirred reactor"),
                scalars=(
                    ScalarConditionSpec(
                        claim_id="initial_temperature",
                        label_quote="initial temperature",
                        quantity_kind=units.QuantityKind.TEMPERATURE,
                        value_quote="823",
                        unit_quote="K",
                        value_cell=TableCellGrounding(table_key=_TABLE_KEY, row=0, col=1, inventory=inventory),
                    ),
                ),
            )

        message = str(excinfo.value)
        assert "not PAPER_PDF" in message
        assert "jats_xml" in message  # the actual node kind is named, not just the expected one
        assert "Absent sha" in message  # the reason the refusal is correct, not incidental


class TestTheDocstringDoesNotAdvertiseTheRefusedArm:
    """Verifier 1, pinned as a tested fact: the ``TableCellGrounding`` docstring must
    not claim a ``MemberSheetKey`` grounding works today, and must name the guard that
    refuses it, so the prose cannot drift back to promising a surface the code rejects.
    Follows the docstring-pinning precedent in ``test_figure_digitization_record``."""

    def test_the_docstring_marks_membersheetkey_reserved_and_names_the_guard(self) -> None:
        doc = TableCellGrounding.__doc__
        assert doc is not None
        # The reserved arm is flagged, not advertised as reachable.
        assert "MemberSheetKey" in doc
        assert "RESERVED" in doc and "NOT YET REACHABLE" in doc
        # The reader is pointed at the guard that enforces it.
        assert "_CellCiter.validate" in doc
        # And told which arm IS reachable today.
        assert "PAPER_PDF" in doc


class TestTheInventoryShaIsAFunctionOfTheGridNotTheKeyOrder:
    """Verifier 2: ``inventory_payload_with_texts`` sorts its (row, col) keys, so the
    same logical grid spelled in two different insertion orders is ONE content address.
    This asserts on identity across orders; before the sort it was FALSE."""

    def test_two_dicts_with_the_same_grid_in_different_orders_share_one_sha(self) -> None:
        raw_sha256 = "a" * 64
        forward = make_embedded_inventory_with_texts(
            raw_sha256=raw_sha256,
            cell_texts={(0, 0): "temperature", (0, 1): "823", (1, 0): "pressure", (1, 1): "1"},
        )
        shuffled = make_embedded_inventory_with_texts(
            raw_sha256=raw_sha256,
            cell_texts={(1, 1): "1", (0, 1): "823", (1, 0): "pressure", (0, 0): "temperature"},
        )
        # Same grid, keys inserted in different orders -> ONE inventory sha.
        assert forward.inventory_sha256 == shuffled.inventory_sha256
