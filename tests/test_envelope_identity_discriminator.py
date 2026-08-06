"""Tests for the identity-payload DISCRIMINATOR shared by both envelope
classes, and for the dataset-side node-order canonicalization.

Two invariants of the identity projection live here rather than in the
per-class identity modules:

1. **Payloads are self-describing.** Every payload projected by
   ``DatasetEnvelope.identity_payload()`` or
   ``ConditionSetEnvelope.identity_payload()`` carries ``envelope_type`` and
   ``identity_payload_version`` in its addressed bytes, and each class's
   ``from_identity_payload`` refuses -- BEFORE model validation -- a payload
   whose discriminator is missing, names the other class, or carries an
   unsupported version. The cross-type refusal is the case that matters
   most: without it, a condition set handed to the dataset parser (or the
   reverse) is rejected only by field-shape accident, which is a property
   of the two schemas' CURRENT fields, not a guarantee, and the resulting
   field-level error would say nothing about the actual problem.

2. **Node tuple order is never identity-bearing.** ``SourceGraph.nodes`` is
   semantically a SET, so the projection sorts nodes by node_id; two
   envelopes differing only in node tuple order must produce identical
   canonical bytes. The condition-set half of this invariant is pinned in
   ``tests/test_dataset_condition_set_identity.py``; the DATASET half lives
   here because ``tests/test_dataset_identity_payload.py`` is the
   golden-pin module, whose standing rule forbids edits beyond its pin --
   its own maximal fixture is constructed already-sorted (to keep the
   positional completeness walker aligned) and so never exercises the
   unsorted case itself.

Cross-module fixture imports (``_maximal_envelope``,
``_maximal_condition_set_envelope``) follow the established convention in
this suite -- see ``tests/test_dataset_bridge.py`` -- rather than copying
fixtures that would silently drift from the projections they exercise.
"""

from __future__ import annotations

import copy

import pytest

from carmel.schemas.datasets import (
    ConditionSetEnvelope,
    DatasetEnvelope,
    DatasetEnvelopeParseError,
    SourceGraph,
)
from carmel.services.dataset_store import canonical_json_bytes
from tests.test_dataset_condition_set_identity import _maximal_condition_set_envelope
from tests.test_dataset_identity_payload import _maximal_envelope

# --------------------------------------------------------------------------
# 1. The discriminator is part of the addressed bytes, for both classes
# --------------------------------------------------------------------------


class TestDiscriminatorIsProjected:
    """If the discriminator were only checked on the way in but never
    actually emitted, every payload would fail the missing-key refusal and
    nothing stored would be self-describing -- these tests pin that the
    keys really appear in the projected payload, with the exact per-class
    values the parsers gate on."""

    def test_dataset_payload_carries_its_own_type_and_version(self) -> None:
        payload = _maximal_envelope().identity_payload()
        assert payload["envelope_type"] == "dataset"
        assert payload["identity_payload_version"] == 1

    def test_condition_set_payload_carries_its_own_type_and_version(self) -> None:
        payload = _maximal_condition_set_envelope().identity_payload()
        assert payload["envelope_type"] == "condition_set"
        assert payload["identity_payload_version"] == 1


# --------------------------------------------------------------------------
# 2. Round trips stay exact WITH the discriminator in the payload
# --------------------------------------------------------------------------


class TestRoundTripRemainsExactWithDiscriminator:
    """The two-stage parse re-projects the parsed envelope and compares
    byte-for-byte against the input. The re-projected payload now contains
    the discriminator too, so these tests prove the comparison stayed exact
    end to end -- i.e. the parser strips the keys before ``model_validate``
    and the projector puts identical ones back."""

    def test_dataset_round_trip_is_byte_exact(self) -> None:
        payload = _maximal_envelope().identity_payload()
        parsed = DatasetEnvelope.from_identity_payload(payload)
        assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(payload)

    def test_condition_set_round_trip_is_byte_exact(self) -> None:
        payload = _maximal_condition_set_envelope().identity_payload()
        parsed = ConditionSetEnvelope.from_identity_payload(payload)
        assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(payload)


# --------------------------------------------------------------------------
# 3. Cross-type refusal -- the case this discriminator exists for
# --------------------------------------------------------------------------


class TestCrossTypeRefusal:
    """A condition-set payload must never be parsed as a dataset, and vice
    versa. Both refusals must fire on the DISCRIMINATOR, before model
    validation, and the message must name what was expected and what was
    found -- a field-level ValidationError here would mean the gate never
    ran and the refusal is back to being a field-shape accident."""

    def test_a_condition_set_payload_handed_to_the_dataset_parser_is_refused(self) -> None:
        payload = _maximal_condition_set_envelope().identity_payload()
        with pytest.raises(DatasetEnvelopeParseError, match="'condition_set'") as excinfo:
            DatasetEnvelope.from_identity_payload(payload)
        assert "'dataset'" in str(excinfo.value), "the refusal must name the expected type too"
        assert "failed validation" not in str(excinfo.value), (
            "the wrong-type payload was refused by field validation, not by the discriminator "
            "gate -- the type mismatch was laundered into a field-shape error"
        )

    def test_a_dataset_payload_handed_to_the_condition_set_parser_is_refused(self) -> None:
        payload = _maximal_envelope().identity_payload()
        with pytest.raises(DatasetEnvelopeParseError, match="'dataset'") as excinfo:
            ConditionSetEnvelope.from_identity_payload(payload)
        assert "'condition_set'" in str(excinfo.value), "the refusal must name the expected type too"
        assert "failed validation" not in str(excinfo.value), (
            "the wrong-type payload was refused by field validation, not by the discriminator "
            "gate -- the type mismatch was laundered into a field-shape error"
        )


# --------------------------------------------------------------------------
# 4. Missing or unsupported discriminators are refused, for both classes
# --------------------------------------------------------------------------

_PARSERS = [
    pytest.param(DatasetEnvelope, _maximal_envelope, id="dataset"),
    pytest.param(ConditionSetEnvelope, _maximal_condition_set_envelope, id="condition-set"),
]


class TestMissingOrUnsupportedDiscriminatorIsRefused:
    """Each refusal is tested per class: a payload that does not say what
    it is, or that was projected under a version this module has never
    seen, cannot be trusted to mean what the parser would make of it."""

    @pytest.mark.parametrize(("envelope_class", "build"), _PARSERS)
    def test_a_payload_with_no_envelope_type_is_refused(self, envelope_class, build) -> None:
        payload = copy.deepcopy(build().identity_payload())
        del payload["envelope_type"]
        with pytest.raises(DatasetEnvelopeParseError, match="no 'envelope_type' key"):
            envelope_class.from_identity_payload(payload)

    @pytest.mark.parametrize(("envelope_class", "build"), _PARSERS)
    def test_a_payload_with_no_version_is_refused(self, envelope_class, build) -> None:
        payload = copy.deepcopy(build().identity_payload())
        del payload["identity_payload_version"]
        with pytest.raises(DatasetEnvelopeParseError, match="no 'identity_payload_version' key"):
            envelope_class.from_identity_payload(payload)

    @pytest.mark.parametrize(("envelope_class", "build"), _PARSERS)
    def test_a_payload_with_an_unsupported_version_is_refused(self, envelope_class, build) -> None:
        payload = copy.deepcopy(build().identity_payload())
        payload["identity_payload_version"] = 2
        with pytest.raises(DatasetEnvelopeParseError, match="supports exactly version 1"):
            envelope_class.from_identity_payload(payload)

    @pytest.mark.parametrize(("envelope_class", "build"), _PARSERS)
    def test_a_boolean_true_version_is_refused_despite_equaling_one(self, envelope_class, build) -> None:
        """``True == 1`` in Python, but no version of this projector ever
        wrote ``true`` into the version slot -- equality alone would wave
        this payload through the gate and leave it to the byte comparison
        to refuse for an unrelated-sounding reason."""
        payload = copy.deepcopy(build().identity_payload())
        payload["identity_payload_version"] = True
        with pytest.raises(DatasetEnvelopeParseError, match="supports exactly version 1"):
            envelope_class.from_identity_payload(payload)


# --------------------------------------------------------------------------
# 5. Dataset-side node-order invariance
# --------------------------------------------------------------------------


class TestDatasetNodeOrderDoesNotAffectIdentity:
    """Two ``DatasetEnvelope``s differing ONLY in node tuple order must
    produce identical canonical bytes. Before the projection sorted nodes
    by node_id, this exact construction produced two different
    ``compute_dataset_sha`` values for one graph -- one dataset with many
    content addresses, which silently defeats write-once dedup and makes
    byte-level payload comparison meaningless."""

    def test_two_node_tuple_orders_produce_identical_canonical_bytes(self) -> None:
        baseline = _maximal_envelope()
        permuted_nodes = tuple(reversed(baseline.source_graph.nodes))
        assert [node.node_id for node in permuted_nodes] != [
            node.node_id for node in baseline.source_graph.nodes
        ], "fixture drift: the permutation must actually change the tuple order"
        permuted = DatasetEnvelope(
            source_graph=SourceGraph(nodes=permuted_nodes),
            composition=baseline.composition,
            series=baseline.series,
            conversion_tables=baseline.conversion_tables,
        )

        assert canonical_json_bytes(permuted.identity_payload()) == canonical_json_bytes(
            baseline.identity_payload()
        ), (
            "two DatasetEnvelopes differing ONLY in node tuple order produced different "
            "canonical bytes -- one dataset holds many content addresses"
        )

    def test_the_projected_nodes_are_sorted_by_node_id(self) -> None:
        """The invariance above could be satisfied by any canonical order;
        this pins WHICH order is canonical, so the golden pin's node
        ordering is explainable from the projection alone."""
        payload = _maximal_envelope().identity_payload()
        projected_ids = [node["node_id"] for node in payload["source_graph"]["nodes"]]
        assert projected_ids == sorted(projected_ids)
