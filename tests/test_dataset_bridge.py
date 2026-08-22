"""Tests for ``DatasetEnvelope.from_identity_payload`` (the explicit inverse
of ``identity_payload()``) and the typed store/load wrappers in
``carmel.services.dataset_bridge``.

Reuses the maximal-envelope fixture machinery from
``tests.test_dataset_identity_payload`` rather than hand-rolling a second one
-- that file already worked out (empirically, via ValidationError message
text) every cross-field invariant needed to construct a ``DatasetEnvelope``
that reaches every field/union-arm this schema can legally expose, and a
second maximal fixture here would just be a second place for those invariants
to silently drift out of sync.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    DatasetEnvelope,
    DatasetEnvelopeParseError,
    SourceGraph,
)
from carmel.services.dataset_bridge import (
    UnstorableDatasetEnvelopeError,
    load_dataset_envelope,
    store_dataset_envelope,
)
from carmel.services.dataset_store import canonical_json_bytes, compute_dataset_sha
from tests.test_dataset_identity_payload import (
    _maximal_bbox,
    _maximal_envelope,
    _minimal_envelope_with_composition,
)


def test_round_trip_on_maximal_envelope() -> None:
    """The maximal envelope survives payload -> parse -> payload unchanged.

    Deliberately does NOT assert ``parsed == envelope``: fields excluded from
    identity on purpose (e.g. ``ArchiveOrigin.member_display_path``, per
    ``_UNADDRESSED_FIELDS``) are real divergences between the parsed envelope
    and the original model instance, not round-trip bugs -- identity is
    defined by ``identity_payload()`` alone, which is exactly what
    ``from_identity_payload`` promises to reproduce.
    """
    envelope = _maximal_envelope()
    payload = envelope.identity_payload()

    parsed = DatasetEnvelope.from_identity_payload(payload)

    assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(payload)


def test_store_and_load_round_trip_by_sha(tmp_path: Path) -> None:
    """store_dataset_envelope + load_dataset_envelope round-trips through the
    actual content-addressed store, keyed by the same sha256 the plain-dict
    store API would compute."""
    envelope = _maximal_envelope()
    payload = envelope.identity_payload()
    expected_sha = compute_dataset_sha(payload)

    stored = store_dataset_envelope(tmp_path, envelope)
    assert stored.sha256 == expected_sha

    loaded = load_dataset_envelope(tmp_path, stored.sha256)

    assert canonical_json_bytes(loaded.identity_payload()) == canonical_json_bytes(payload)


def test_absent_with_note_deep_in_tree_survives_round_trip() -> None:
    """An Absent marker carrying a non-None note, nested several levels deep
    (DatasetEnvelope.composition -> Absent), must rehydrate to an equal
    Absent instance, not merely "some Absent"."""
    composition = Absent(reason=AbsenceReason.CONFLICTING_SOURCES, note="sentinel")
    envelope = _minimal_envelope_with_composition(composition)
    payload = envelope.identity_payload()
    assert payload["composition"] == {
        "__absent__": True,
        "reason": "conflicting_sources",
        "note": "sentinel",
    }

    parsed = DatasetEnvelope.from_identity_payload(payload)

    assert parsed.composition == composition
    assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(payload)


def test_present_looking_dict_without_marker_is_rejected_at_stage_two() -> None:
    """A dict that looks like a present ArchiveOrigin but is actually missing
    its required field can validate as `Absent` under pydantic's "smart"
    union mode once the (deliberately absent) `__absent__` key is
    disregarded -- `model_validate` alone would accept this silently. Only
    the stage-2 round-trip byte comparison in `from_identity_payload` catches
    it, because the re-projected Absent adds back `__absent__: True`, which
    the mutated input never had.

    This is a value observed empirically by running the parser, not merely
    reasoned about: `_rehydrate_identity_payload` leaves the mutated dict
    alone (no `__absent__` key, so it is not recognized as a marker in stage
    1), `model_validate` accepts it as `Absent(reason=UNKNOWN, note=None)` (no
    ValidationError raised), and the raised error is the round-trip
    DatasetEnvelopeParseError, not a validation failure -- confirmed below by
    asserting the error message names it as a round-trip disagreement.
    """
    envelope = _maximal_envelope()
    payload = copy.deepcopy(envelope.identity_payload())

    si_node = next(node for node in payload["source_graph"]["nodes"] if node["node_id"] == "si")
    assert si_node["origin"] == {"archive_sha256": "b" * 64}
    si_node["origin"] = {"reason": "unknown", "note": None}

    with pytest.raises(DatasetEnvelopeParseError, match="does not byte-match"):
        DatasetEnvelope.from_identity_payload(payload)


@pytest.mark.parametrize(
    "mutate_marker",
    [
        pytest.param(lambda marker: marker.update(__absent__=False), id="absent_key_false"),
        # A TRUTHY-but-not-True `__absent__` must be rejected exactly as hard as
        # a falsy one. Without this case, `is not True` can be loosened to a
        # plain truthiness test and every other test in this file still passes
        # -- mutation-proved, which is why the case is here. The strictness is
        # the point: the projector emits the literal `True` and nothing else, so
        # anything else in that slot means the payload did not come from this
        # projector, and guessing what a `1` was meant to mean is precisely the
        # silent reinterpretation stage 1 exists to refuse.
        pytest.param(lambda marker: marker.update(__absent__=1), id="absent_key_truthy_not_true"),
        pytest.param(lambda marker: marker.update(extra_key="surprise"), id="extra_key"),
        pytest.param(lambda marker: marker.update(reason="not_a_real_reason"), id="unknown_reason"),
    ],
)
def test_malformed_absence_marker_is_rejected_at_stage_one(
    mutate_marker: Callable[[dict[str, Any]], None],
) -> None:
    """Malformed `__absent__` markers are rejected during rehydration
    (stage 1), before `model_validate` is ever called -- the mere presence
    of the `__absent__` key is enough to commit to marker-shape validation,
    so a bad shape can never fall through and be silently reinterpreted as
    ordinary present-value data.
    """
    envelope = _maximal_envelope()
    payload = copy.deepcopy(envelope.identity_payload())

    jats_node = next(node for node in payload["source_graph"]["nodes"] if node["node_id"] == "jats")
    assert jats_node["origin"]["__absent__"] is True
    mutate_marker(jats_node["origin"])

    with pytest.raises(DatasetEnvelopeParseError) as excinfo:
        DatasetEnvelope.from_identity_payload(payload)

    # Asserting the ERROR TYPE alone would not test what this test claims to
    # test: BOTH stages raise DatasetEnvelopeParseError, so a stage-1 check
    # that stopped firing would simply fall through to stage 2's round-trip
    # comparison -- which rejects these same payloads anyway, for a different
    # reason -- and this test would stay green while no longer testing stage 1
    # at all. Mutation-proved: loosening `is not True` to a plain truthiness
    # test let `__absent__=1` through stage 1, and every assertion here still
    # passed until this line was added.
    #
    # "does not byte-match" is the stage-2 message and appears nowhere in any
    # stage-1 message, so excluding it pins that stage 1 is what rejected.
    assert "does not byte-match" not in str(excinfo.value), (
        "this payload was rejected by the stage-2 round-trip check, not by stage-1 marker validation -- "
        "the marker-shape check has stopped firing and this test is no longer testing it"
    )


def test_reversed_series_order_is_rejected_by_model_validate() -> None:
    """Reversing the canonically-sorted `series` list violates E1b
    (`_validate_series_sorted`), which fires inside `model_validate` itself
    during stage 2 -- i.e. as an ordinary pydantic ValidationError, wrapped
    into DatasetEnvelopeParseError, well before the round-trip byte
    comparison would ever run.
    """
    envelope = _maximal_envelope()
    payload = copy.deepcopy(envelope.identity_payload())
    assert [series["series_id"] for series in payload["series"]] == ["s1", "s2"]
    payload["series"].reverse()

    with pytest.raises(DatasetEnvelopeParseError, match="failed validation"):
        DatasetEnvelope.from_identity_payload(payload)


def test_corrupted_char_span_locator_end_is_rejected() -> None:
    """A CharSpanLocator with end <= start violates the schema's own
    `_validate_span_nonempty` invariant, firing inside `model_validate`
    during stage 2 (a pydantic ValidationError), not the round-trip check.
    """
    envelope = _maximal_envelope()
    payload = copy.deepcopy(envelope.identity_payload())

    locators = _find_locators(payload, kind="char_span")
    assert locators, "fixture must contain at least one CharSpanLocator"
    locator = locators[0]
    assert locator["end"] > locator["start"]
    locator["end"] = locator["start"]

    with pytest.raises(DatasetEnvelopeParseError, match="failed validation"):
        DatasetEnvelope.from_identity_payload(payload)


def test_corrupted_char_span_locator_text_space_is_rejected() -> None:
    """An unrecognized `text_space` value on a CharSpanLocator is rejected by
    ordinary pydantic enum validation during `model_validate` (stage 2).
    """
    envelope = _maximal_envelope()
    payload = copy.deepcopy(envelope.identity_payload())

    locators = _find_locators(payload, kind="char_span")
    assert locators
    locators[0]["text_space"] = "not_a_real_text_space"

    with pytest.raises(DatasetEnvelopeParseError, match="failed validation"):
        DatasetEnvelope.from_identity_payload(payload)


def _find_locators(payload: dict[str, Any], *, kind: str) -> list[dict[str, Any]]:
    """Walk a DatasetEnvelope identity payload looking for every
    ``source_ref.locator`` dict of the given ``kind``. Small ad hoc walker
    local to this test module -- the payload shape is deeply nested and
    there is no reusable locator-finder elsewhere in the codebase.
    """
    found: list[dict[str, Any]] = []

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("kind") == kind:
                found.append(value)
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(payload)
    return found


def test_round_trip_drops_unaddressed_display_only_fields() -> None:
    """PIN: the round trip is exact on IDENTITY and lossy on everything else.

    `ArchiveOrigin.member_display_path` is the one field registered in
    `_UNADDRESSED_FIELDS`, so `identity_payload()` never emits it and
    `from_identity_payload()` cannot bring it back. An envelope stored with a
    display path loads back with `None` there, and no error is raised --
    correctly, because the guarantee is "addresses the same dataset", not
    "is the same object".

    This is pinned rather than merely documented for two reasons. First, the
    natural round-trip assertion to reach for is `parsed == envelope`, which
    is FALSE here; without this test the next person writes it, watches it
    fail, and has to rediscover why. Second, and more important: if anyone
    later projects `member_display_path` into the identity payload, this test
    fails and forces the question out into the open -- because doing so
    changes the sha of every dataset whose graph carries an SI member, i.e.
    silently re-addresses already-stored data. That is exactly the hazard
    `compute_dataset_sha`'s plain-dict contract exists to prevent.
    """
    envelope = _maximal_envelope()

    origins = [node.origin for node in envelope.source_graph.nodes if not isinstance(node.origin, Absent)]
    assert origins, "fixture regression: the maximal envelope no longer has any ArchiveOrigin to test"
    assert any(origin.member_display_path is not None for origin in origins), (
        "fixture regression: no ArchiveOrigin carries a member_display_path, so this test proves nothing"
    )

    parsed = DatasetEnvelope.from_identity_payload(envelope.identity_payload())

    parsed_origins = [node.origin for node in parsed.source_graph.nodes if not isinstance(node.origin, Absent)]
    assert len(parsed_origins) == len(origins)
    assert all(origin.member_display_path is None for origin in parsed_origins), (
        "member_display_path came back from a round trip -- it is registered in _UNADDRESSED_FIELDS, so "
        "identity_payload() must not be emitting it; if that changed deliberately, every stored dataset "
        "whose graph carries an SI member just got a new sha"
    )

    # The identity, meanwhile, is preserved exactly -- that is the whole
    # contract. Asserting both halves in one test keeps them from drifting
    # apart: a change that "fixed" the lossiness by projecting the field
    # would satisfy neither.
    assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(envelope.identity_payload())
    assert parsed != envelope


class TestStoreRefusesAnEnvelopeItsOwnLoaderWouldReject:
    """The store is WRITE-ONCE and IMMUTABLE, so an unreadable write is forever.

    A typed ``DatasetEnvelope`` in hand is not proof it was ever validated:
    ``model_construct`` builds one with every validator skipped. Probed before
    the fix, such an envelope stored happily and ``load_dataset_envelope``
    then refused it permanently -- an address burned on bytes nothing can
    read, in a store whose whole contract is that writes are final.
    """

    @staticmethod
    def _unvalidated_envelope() -> DatasetEnvelope:
        """An envelope that skipped validation and violates ``series`` MinLen(1).

        Built from a REAL maximal envelope so that everything except the one
        broken invariant is genuine -- a hand-rolled stub could fail to store
        for some unrelated reason and the test would pass while proving
        nothing about the refusal under test.
        """
        valid = _maximal_envelope()
        return DatasetEnvelope.model_construct(**{**valid.__dict__, "series": ()})

    def test_store_refuses_what_load_could_never_read_back(self, tmp_path: Path) -> None:
        envelope = self._unvalidated_envelope()
        assert envelope.series == ()  # the validators really were skipped
        with pytest.raises(UnstorableDatasetEnvelopeError) as excinfo:
            store_dataset_envelope(tmp_path, envelope)
        # The refusal must say what the reader would have choked on, not just
        # "invalid" -- the caller cannot fix what the message does not name.
        assert "at least 1 item" in str(excinfo.value)

    @staticmethod
    def _envelope_with_a_stray_crop_region() -> DatasetEnvelope:
        """A PAPER_PDF carrying a real ``BBox`` -- illegal under I7, so
        reachable only by skipping validation.

        The interesting case for a CONDITIONALLY projected field, and the
        reason ``_addresses_a_crop_region`` reads the field's VALUE rather
        than the node's ``kind``. Value-keyed, this node projects its stray
        region, the payload no longer parses, and the write is refused.
        Kind-keyed, the projection would drop it, the tampered envelope would
        address byte-identically to a clean one, and the store would take it
        -- nothing untrue written, but a producer bug gone unremarked.
        """
        valid = _maximal_envelope()
        paper = valid.source_graph.node("paper")
        tampered = paper.model_copy()
        object.__setattr__(tampered, "crop_region", _maximal_bbox())
        nodes = tuple(tampered if node.node_id == "paper" else node for node in valid.source_graph.nodes)
        graph = SourceGraph.model_construct(nodes=nodes)
        return DatasetEnvelope.model_construct(**{**valid.__dict__, "source_graph": graph})

    def test_store_refuses_a_node_carrying_a_region_it_has_no_right_to(self, tmp_path: Path) -> None:
        envelope = self._envelope_with_a_stray_crop_region()
        assert not isinstance(envelope.source_graph.node("paper").crop_region, Absent), (
            "fixture drift: the validators were supposed to be skipped"
        )

        with pytest.raises(UnstorableDatasetEnvelopeError) as excinfo:
            store_dataset_envelope(tmp_path, envelope)

        assert "crop_region" in str(excinfo.value), (
            "the refusal must name what the reader would choke on -- if this fails, the projection "
            "dropped the stray region and the tampered envelope was stored, or refused for some "
            "unrelated reason"
        )

    def test_the_stray_region_does_not_address_like_a_clean_envelope(self) -> None:
        """The half that pins the design rather than the refusal: the tampered
        envelope must not project the SAME bytes as a clean one. If it did,
        the refusal above would evaporate the moment some neighbouring
        invariant moved, and nothing would notice it had."""
        tampered = canonical_json_bytes(self._envelope_with_a_stray_crop_region().identity_payload())

        assert tampered != canonical_json_bytes(_maximal_envelope().identity_payload())

    def test_the_refusal_writes_nothing_at_all(self, tmp_path: Path) -> None:
        """Fail CLOSED, not fail-halfway.

        A refusal that had already written bytes would be worse than no check:
        it would burn the address AND raise, leaving a caller that retries
        unable to ever succeed at that sha.
        """
        before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        with pytest.raises(UnstorableDatasetEnvelopeError):
            store_dataset_envelope(tmp_path, self._unvalidated_envelope())
        after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
        assert after == before

    def test_a_valid_envelope_still_stores_at_its_unchanged_address(self, tmp_path: Path) -> None:
        """The counterweight, and it is doing real work.

        Without it, a `raise` on EVERY store would satisfy both tests above.
        It also pins that adding the check did not disturb the canonical bytes:
        the address must still be exactly ``compute_dataset_sha(payload)``, the
        value the schema-blind store computes with no knowledge of this module.
        """
        envelope = _maximal_envelope()
        payload = envelope.identity_payload()
        stored = store_dataset_envelope(tmp_path, envelope)
        assert stored.sha256 == compute_dataset_sha(payload)
        # Compared as canonical bytes, not with ``==``: identity in this store
        # IS the canonical rendering, and model equality answers a different
        # question (it is not structural over every nested field here).
        loaded = load_dataset_envelope(tmp_path, stored.sha256)
        assert canonical_json_bytes(loaded.identity_payload()) == canonical_json_bytes(payload)

    def test_a_refused_write_is_not_reported_as_a_corrupt_read(self) -> None:
        """``UnstorableDatasetEnvelopeError`` must NOT be a
        ``DatasetEnvelopeParseError``.

        The two say opposite things about the store: a parse error means bytes
        on disk are bad (a fact about its past); this means bytes were never
        written (a fact about a refused future). If the refusal subclassed the
        parse error, a caller with an ``except DatasetEnvelopeParseError``
        around a read-repair path would silently swallow a rejected write and
        believe the store had a corruption problem rather than that its own
        envelope was invalid.
        """
        assert not issubclass(UnstorableDatasetEnvelopeError, DatasetEnvelopeParseError)
        assert not issubclass(DatasetEnvelopeParseError, UnstorableDatasetEnvelopeError)

    def test_the_check_examines_the_payload_it_will_write_not_the_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validating the OBJECT would be a different, weaker check.

        ``identity_payload()`` -> ``from_identity_payload()`` is a projection
        and a rehydration, not an identity function. An envelope whose object
        validates but whose PAYLOAD will not rehydrate is precisely the case
        that burns an address, and object-validation cannot see it. No such
        asymmetry exists in the schema today, so the two are separated here by
        patching the projection -- the only way to tell which artifact the
        check actually looks at.

        This test was written because a mutation audit found that replacing
        the payload round-trip with ``model_validate(envelope.model_dump())``
        passed every other test in this class.
        """
        envelope = _maximal_envelope()
        good_payload = envelope.identity_payload()
        monkeypatch.setattr(
            DatasetEnvelope,
            "identity_payload",
            lambda self: {k: v for k, v in good_payload.items() if k != "series"},
        )
        with pytest.raises(UnstorableDatasetEnvelopeError):
            store_dataset_envelope(tmp_path, envelope)
        assert sorted(tmp_path.rglob("*")) == []
