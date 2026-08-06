# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed store/load bridge over ``ConditionSetEnvelope``.

Mirrors ``tests.test_dataset_bridge`` in shape, because the two bridges make
the same promise about the same write-once store. What is NEW here, and what
most of this module is about, is that there are now TWO envelope types sharing
one store: the failure this file exists to prevent is one type being written or
read through the other type's door.

Reuses the maximal-envelope fixture from
``tests.test_dataset_condition_set_identity`` rather than hand-rolling a second
one -- that module already worked out every cross-field invariant needed to
build a ``ConditionSetEnvelope`` that reaches every field and both subject
union arms, and a second fixture here would just be a second place for those
invariants to drift.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from carmel.schemas.datasets import (
    _CONDITION_SET_ENVELOPE_TYPE,
    _ENVELOPE_TYPE_KEY,
    _IDENTITY_PAYLOAD_VERSION_KEY,
    ConditionSetEnvelope,
    DatasetEnvelope,
    DatasetEnvelopeParseError,
)
from carmel.services._envelope_bridge import TypedEnvelopeSpec, store_typed_envelope
from carmel.services.condition_set_bridge import (
    UnstorableConditionSetEnvelopeError,
    load_condition_set_envelope,
    store_condition_set_envelope,
)
from carmel.services.dataset_bridge import (
    UnstorableDatasetEnvelopeError,
    load_dataset_envelope,
    store_dataset_envelope,
)
from carmel.services.dataset_store import (
    CONDITION_SET_STORE_DIR,
    DATASET_STORE_DIR,
    canonical_json_bytes,
    compute_dataset_sha,
    dataset_decimal_repr_version,
    list_datasets,
    store_dataset,
    verify_dataset,
)
from tests.test_dataset_condition_set_identity import (
    _maximal_condition_set_envelope,
    _unresolved_subject,
)
from tests.test_dataset_identity_payload import _maximal_envelope


class TestOneTypeCannotBeWrittenThroughTheOtherTypesDoor:
    """The reason this bridge is not one generic function with a ``cls=``.

    ``e8c039e`` gave both envelopes an ``envelope_type`` discriminator and both
    parsers a refusal, so a condition set routed through the dataset bridge is
    already refused -- but only by accident of the PARSER, downstream of the
    decision. These tests pin it as a property of the BRIDGE, so that the
    refusal survives someone later "simplifying" the parsers, and so that the
    two directions are pinned symmetrically rather than one being an untested
    mirror of the other.

    Both calls below are deliberate type violations, hence the ``type: ignore``.
    That is the point: mypy is not present at runtime, this repo already has
    call sites that build envelopes via ``model_construct`` with validators
    skipped, and a write to a write-once immutable store is exactly the wrong
    place to rely on a static check nobody runs in production.
    """

    def test_a_condition_set_is_refused_by_the_dataset_bridge(self, tmp_path: Path) -> None:
        envelope = _maximal_condition_set_envelope()
        with pytest.raises(UnstorableDatasetEnvelopeError) as excinfo:
            store_dataset_envelope(tmp_path, envelope)  # type: ignore[arg-type]
        # The message must name the type confusion AND the door the caller
        # should have used, not merely "invalid" -- a caller that cannot see
        # WHICH type it got wrong will retry the same wrong call.
        #
        # Asserting the function name is also what makes the bridge's own
        # discriminator check independently killable: the parser downstream
        # refuses this payload too, and its message happens to contain both
        # type names, so an assertion on the type names alone would still pass
        # with the bridge-level check deleted.
        message = str(excinfo.value)
        assert "condition_set" in message
        assert "dataset" in message
        assert "store_dataset_envelope" in message

    def test_a_dataset_is_refused_by_the_condition_set_bridge(self, tmp_path: Path) -> None:
        envelope = _maximal_envelope()
        with pytest.raises(UnstorableConditionSetEnvelopeError) as excinfo:
            store_condition_set_envelope(tmp_path, envelope)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "condition_set" in message
        assert "dataset" in message
        assert "store_condition_set_envelope" in message

    def test_neither_refused_write_leaves_bytes_behind(self, tmp_path: Path) -> None:
        """Fail CLOSED. A refusal that had already written would burn an
        address AND raise, so a retry could never succeed at that sha."""
        with pytest.raises(UnstorableDatasetEnvelopeError):
            store_dataset_envelope(tmp_path, _maximal_condition_set_envelope())  # type: ignore[arg-type]
        with pytest.raises(UnstorableConditionSetEnvelopeError):
            store_condition_set_envelope(tmp_path, _maximal_envelope())  # type: ignore[arg-type]
        assert sorted(tmp_path.rglob("*")) == []

    def test_a_condition_set_hand_planted_in_the_dataset_dir_is_refused_on_read(
        self, tmp_path: Path
    ) -> None:
        """The read-side mirror, and it is not hypothetical.

        The bridges cannot stop a payload reaching the dataset directory by
        some other route -- a hand-copied file, a restored backup, a future
        caller of the schema-blind ``store_dataset`` directly. Whatever put it
        there, ``load_dataset_envelope`` must refuse to reinterpret it as a
        dataset rather than returning a wrong-typed object.
        """
        payload = _maximal_condition_set_envelope().identity_payload()
        stored = store_dataset(tmp_path, payload)  # the schema-blind door, no type check
        with pytest.raises(DatasetEnvelopeParseError):
            load_dataset_envelope(tmp_path, stored.sha256)

    def test_a_dataset_hand_planted_in_the_condition_set_dir_is_refused_on_read(
        self, tmp_path: Path
    ) -> None:
        payload = _maximal_envelope().identity_payload()
        stored = store_dataset(tmp_path, payload, store_dir=CONDITION_SET_STORE_DIR)
        with pytest.raises(DatasetEnvelopeParseError):
            load_condition_set_envelope(tmp_path, stored.sha256)


class TestTheRoundTripThatMakesTheRefusalsMeanSomething:
    """Counterweight to the class above, and it is doing real work: a bridge
    that raised on EVERY store would satisfy every refusal test written here.
    """

    def test_store_and_load_round_trip_by_sha(self, tmp_path: Path) -> None:
        envelope = _maximal_condition_set_envelope()
        payload = envelope.identity_payload()

        stored = store_condition_set_envelope(tmp_path, envelope)

        # The address must be exactly what the schema-blind store computes with
        # no knowledge of this module -- the bridge adds a projection and a
        # refusal, never an address.
        assert stored.sha256 == compute_dataset_sha(payload)
        loaded = load_condition_set_envelope(tmp_path, stored.sha256)
        assert isinstance(loaded, ConditionSetEnvelope)
        # Compared as canonical bytes, not with ``==``: identity in this store
        # IS the canonical rendering, and model equality answers a different
        # question.
        assert canonical_json_bytes(loaded.identity_payload()) == canonical_json_bytes(payload)

    def test_both_subject_variants_round_trip(self, tmp_path: Path) -> None:
        """The subject is a SUM, and a sum is exactly where a bridge silently
        collapses one arm into the other. Pin both arms through real bytes."""
        for envelope in (
            _maximal_condition_set_envelope(),
            _maximal_condition_set_envelope(subject=_unresolved_subject()),
        ):
            stored = store_condition_set_envelope(tmp_path, envelope)
            loaded = load_condition_set_envelope(tmp_path, stored.sha256)
            assert type(loaded.subject) is type(envelope.subject)
            assert canonical_json_bytes(loaded.identity_payload()) == canonical_json_bytes(
                envelope.identity_payload()
            )

    def test_storing_twice_is_idempotent(self, tmp_path: Path) -> None:
        envelope = _maximal_condition_set_envelope()
        first = store_condition_set_envelope(tmp_path, envelope)
        second = store_condition_set_envelope(tmp_path, envelope)
        assert first == second


class TestConditionSetsLiveInTheirOwnDirectory:
    """Separate directories are not cosmetic here.

    The user chose "query by scanning files" as the query model for this store,
    which makes the DIRECTORY the type index. A condition set sitting in
    ``evidence/datasets/`` would be handed to every dataset consumer that
    enumerates the store, and each would have to re-derive the type from the
    payload to skip it.
    """

    def test_a_stored_condition_set_lands_under_the_condition_set_dir(
        self, tmp_path: Path
    ) -> None:
        stored = store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert stored.path.is_relative_to(tmp_path / CONDITION_SET_STORE_DIR)
        assert not stored.path.is_relative_to(tmp_path / DATASET_STORE_DIR)

    def test_list_datasets_does_not_report_a_condition_set(self, tmp_path: Path) -> None:
        store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert list_datasets(tmp_path) == []

    def test_a_dataset_and_a_condition_set_coexist_without_shadowing(
        self, tmp_path: Path
    ) -> None:
        """Both still load as themselves once both are present."""
        dataset = _maximal_envelope()
        condition_set = _maximal_condition_set_envelope()
        stored_dataset = store_dataset_envelope(tmp_path, dataset)
        stored_condition_set = store_condition_set_envelope(tmp_path, condition_set)

        assert list_datasets(tmp_path) == [stored_dataset.sha256]
        assert isinstance(load_dataset_envelope(tmp_path, stored_dataset.sha256), DatasetEnvelope)
        assert isinstance(
            load_condition_set_envelope(tmp_path, stored_condition_set.sha256),
            ConditionSetEnvelope,
        )


class TestTheStoresOwnIntegrityToolsReachTheSecondDirectory:
    """``verify_dataset``/``list_datasets``/``dataset_decimal_repr_version`` are
    the store's integrity surface -- the documented way a caller answers "does
    this sha resolve to genuine, canonical bytes". A second store directory
    those tools cannot see would be a directory whose contents can never be
    audited, in a system whose entire premise is auditability. They take the
    same ``store_dir`` the bridge does; these pin that it actually reaches them.
    """

    def test_a_stored_condition_set_verifies_in_its_own_directory(self, tmp_path: Path) -> None:
        stored = store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert verify_dataset(tmp_path, stored.sha256, store_dir=CONDITION_SET_STORE_DIR) is True

    def test_it_is_not_verifiable_as_a_dataset(self, tmp_path: Path) -> None:
        """The counterweight: ``verify_dataset`` returning True regardless of
        directory would pass the test above while proving nothing."""
        stored = store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert verify_dataset(tmp_path, stored.sha256) is False

    def test_verification_is_type_blind_and_that_limit_is_pinned_here(
        self, tmp_path: Path
    ) -> None:
        """``verify_dataset`` answers "are these bytes canonical and correctly
        named", NOT "is this a genuine condition set".

        A dataset payload hand-planted in the condition-set directory verifies
        True and lists there too. That is correct for what the function
        measures -- the store is schema-blind by design and must stay so -- but
        it is exactly the kind of limit that gets forgotten and then quietly
        relied on. Pinned as an explicit expectation so that a future caller
        reaching for ``verify_dataset`` as a type check finds this test first,
        and so that the day someone DOES add type-aware verification, this test
        fails and forces the decision to be made deliberately.
        """
        payload = _maximal_envelope().identity_payload()
        stored = store_dataset(tmp_path, payload, store_dir=CONDITION_SET_STORE_DIR)

        assert verify_dataset(tmp_path, stored.sha256, store_dir=CONDITION_SET_STORE_DIR) is True
        assert list_datasets(tmp_path, store_dir=CONDITION_SET_STORE_DIR) == [stored.sha256]
        # The typed loader is the ONLY thing that refuses it, which is why a
        # caller needing a type guarantee has to go through one.
        with pytest.raises(DatasetEnvelopeParseError):
            load_condition_set_envelope(tmp_path, stored.sha256)

    def test_an_unrecognized_store_directory_is_refused(self, tmp_path: Path) -> None:
        """A typo'd directory would be a silent third store: the write succeeds
        and the records are then invisible to every correctly spelled reader.
        In an append-only store that is worse than a failed write, so the
        directory is checked against a known set rather than merely resolved
        for containment -- containment only catches a path that ESCAPES the
        workspace, and this one does not.
        """
        payload = _maximal_condition_set_envelope().identity_payload()
        with pytest.raises(ValueError, match="unknown store directory"):
            store_dataset(tmp_path, payload, store_dir="evidence/condition_sets_typo")
        with pytest.raises(ValueError, match="unknown store directory"):
            list_datasets(tmp_path, store_dir="evidence/condition_sets_typo")
        assert sorted(tmp_path.rglob("*")) == []

    def test_condition_sets_are_enumerable_in_their_own_directory(self, tmp_path: Path) -> None:
        stored = store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert list_datasets(tmp_path, store_dir=CONDITION_SET_STORE_DIR) == [stored.sha256]

    def test_the_decimal_repr_version_marker_is_readable_there_too(self, tmp_path: Path) -> None:
        """The marker is what turns a future change to decimal rendering into a
        migration instead of a silent re-addressing. A directory where it
        cannot be read back is a directory that loses that guarantee."""
        stored = store_condition_set_envelope(tmp_path, _maximal_condition_set_envelope())
        assert (
            dataset_decimal_repr_version(tmp_path, stored.sha256, store_dir=CONDITION_SET_STORE_DIR)
            == 1
        )


class TestStoreRefusesAnEnvelopeItsOwnLoaderWouldReject:
    """Same promise the dataset bridge already makes, re-proved for this type.

    A typed ``ConditionSetEnvelope`` in hand is not proof it was ever
    validated: ``model_construct`` builds one with every validator skipped, and
    this store is write-once and IMMUTABLE, so an unreadable write is forever.
    """

    @staticmethod
    def _unvalidated_envelope() -> ConditionSetEnvelope:
        """An envelope that skipped validation: the all-empty condition set.

        Built from a REAL maximal envelope, so everything that is still present
        is genuine -- a hand-rolled stub could fail to store for some unrelated
        reason and this test would pass while proving nothing.

        Which invariant fires first is deliberately NOT asserted below, and
        that took a measurement to learn: C1 (the joint non-emptiness rule) is
        not reachable in isolation from a maximal fixture, because emptying the
        three claim collections also orphans everything those claims grounded
        -- first the embedded conversion table their ``MeasuredValue`` cited,
        then, once that is dropped too, the source-graph nodes their
        ``SourceRef``s targeted. Each is a real invariant catching a real
        incoherence, and an all-empty envelope violates all of them at once.
        That cascade is a property of the schema being coherent, not a defect
        to engineer around, so this test pins the thing the BRIDGE actually
        promises: nothing is written, and the refusal carries the complaint of
        the loader that would have rejected it, named.
        """
        valid = _maximal_condition_set_envelope()
        return ConditionSetEnvelope.model_construct(
            **{
                **valid.__dict__,
                "scalar_claims": (),
                "categorical_claims": (),
                "unextracted": (),
                "conversion_tables": (),
            }
        )

    def test_store_refuses_what_load_could_never_read_back(self, tmp_path: Path) -> None:
        envelope = self._unvalidated_envelope()
        assert envelope.scalar_claims == ()  # the validators really were skipped
        with pytest.raises(UnstorableConditionSetEnvelopeError) as excinfo:
            store_condition_set_envelope(tmp_path, envelope)
        message = str(excinfo.value)
        # The refusal must name WHOSE loader would choke, and pass that
        # loader's own complaint through -- a caller cannot fix what the
        # message does not name, and "invalid" names nothing. Asserting the
        # class rather than a particular invariant's wording keeps this test
        # honest about what the bridge guarantees (see the fixture docstring).
        assert "ConditionSetEnvelope" in message
        assert "validation error for ConditionSetEnvelope" in message

    def test_the_refusal_writes_nothing_at_all(self, tmp_path: Path) -> None:
        with pytest.raises(UnstorableConditionSetEnvelopeError):
            store_condition_set_envelope(tmp_path, self._unvalidated_envelope())
        assert sorted(tmp_path.rglob("*")) == []

    def test_a_refused_write_is_not_reported_as_a_corrupt_read(self) -> None:
        """``UnstorableConditionSetEnvelopeError`` must NOT be a
        ``DatasetEnvelopeParseError``, for the same reason its dataset sibling
        must not be: one says the bytes on disk are bad, the other says bytes
        were never written. A caller with an ``except DatasetEnvelopeParseError``
        read-repair path would otherwise swallow a rejected write.
        """
        assert not issubclass(UnstorableConditionSetEnvelopeError, DatasetEnvelopeParseError)
        assert not issubclass(DatasetEnvelopeParseError, UnstorableConditionSetEnvelopeError)

    def test_the_two_bridges_refusals_are_not_interchangeable(self) -> None:
        """Neither refusal may be caught by an ``except`` written for the
        other. If one subclassed the other, a caller handling a rejected
        dataset write would silently also swallow a rejected condition-set
        write -- which is precisely the type confusion this whole module
        exists to make impossible.
        """
        assert not issubclass(UnstorableConditionSetEnvelopeError, UnstorableDatasetEnvelopeError)
        assert not issubclass(UnstorableDatasetEnvelopeError, UnstorableConditionSetEnvelopeError)

    def test_the_check_examines_the_payload_it_will_write_not_the_object(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validating the OBJECT would be a different, weaker check.

        ``identity_payload()`` -> ``from_identity_payload()`` is a projection
        and a rehydration, not an identity function. An envelope whose object
        validates but whose PAYLOAD will not rehydrate is exactly the case that
        burns an address, and object-validation cannot see it. No such
        asymmetry exists in the schema today, so the two are separated here by
        patching the projection -- the only way to tell which artifact the
        check actually looks at. (The dataset bridge grew this test after a
        mutation audit; the kernel is shared now, so it is pinned from both
        sides.)
        """
        envelope = _maximal_condition_set_envelope()
        good_payload = envelope.identity_payload()
        monkeypatch.setattr(
            ConditionSetEnvelope,
            "identity_payload",
            lambda self: {k: v for k, v in good_payload.items() if k != "subject"},
        )
        with pytest.raises(UnstorableConditionSetEnvelopeError):
            store_condition_set_envelope(tmp_path, envelope)
        assert sorted(tmp_path.rglob("*")) == []


class _NormalizingEnvelope:
    """Satisfies ``AddressableEnvelope`` structurally, and normalizes on the way in.

    A stand-in for the third envelope type nobody has written yet. It is a
    plain class, not a pydantic model, precisely because that is the point: the
    protocol is a STATIC construct, so anything with the two right method names
    passes it, including something whose ``from_identity_payload`` quietly
    drops a key it does not recognise.
    """

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = dict(payload)

    def identity_payload(self) -> dict[str, object]:
        return dict(self._payload)

    @classmethod
    def from_identity_payload(cls, payload: dict[str, object]) -> _NormalizingEnvelope:
        return cls({key: value for key, value in payload.items() if key != "an_unrecognized_key"})


class _FaithfulEnvelope(_NormalizingEnvelope):
    """The same stub, minus the normalization."""

    @classmethod
    def from_identity_payload(cls, payload: dict[str, object]) -> _FaithfulEnvelope:
        return cls(payload)


def _stub_spec(envelope_cls: type[_NormalizingEnvelope]) -> TypedEnvelopeSpec[Any]:
    return TypedEnvelopeSpec(
        envelope_cls=envelope_cls,
        envelope_type=_CONDITION_SET_ENVELOPE_TYPE,
        store_dir=CONDITION_SET_STORE_DIR,
        unstorable_error=UnstorableConditionSetEnvelopeError,
        store_function_name="store_condition_set_envelope",
    )


def _stub_payload() -> dict[str, object]:
    return {
        _ENVELOPE_TYPE_KEY: _CONDITION_SET_ENVELOPE_TYPE,
        _IDENTITY_PAYLOAD_VERSION_KEY: 1,
        "an_unrecognized_key": "dropped by the normalizing variant",
    }


class TestAPayloadThatDoesNotSurviveItsOwnRoundTripIsRefused:
    """The kernel re-projects and compares bytes, and that is NOT redundant.

    Both envelope types check this inside ``from_identity_payload`` today, so
    against the real classes the check can never fire -- which is exactly how a
    guard like this rots into an unkillable line that everyone assumes is
    working. It defends the case the type system cannot: ``AddressableEnvelope``
    is a ``Protocol`` and constrains nothing at runtime, so a future class that
    accepts-and-normalizes a payload would have its PRE-parse bytes written
    under an address no reader could ever reproduce. Permanently, in a
    write-once store.

    Testing it needs a class that does not exist yet, so these build one.
    """

    def test_a_normalizing_rehydration_is_refused(self, tmp_path: Path) -> None:
        spec = _stub_spec(_NormalizingEnvelope)
        with pytest.raises(UnstorableConditionSetEnvelopeError) as excinfo:
            store_typed_envelope(spec, tmp_path, _NormalizingEnvelope(_stub_payload()))
        assert "does not survive its own round trip" in str(excinfo.value)

    def test_the_refusal_writes_nothing(self, tmp_path: Path) -> None:
        spec = _stub_spec(_NormalizingEnvelope)
        with pytest.raises(UnstorableConditionSetEnvelopeError):
            store_typed_envelope(spec, tmp_path, _NormalizingEnvelope(_stub_payload()))
        assert sorted(tmp_path.rglob("*")) == []

    def test_a_faithful_rehydration_still_stores(self, tmp_path: Path) -> None:
        """The counterweight, and it is load-bearing: a kernel that refused
        EVERY store would satisfy both tests above."""
        spec = _stub_spec(_FaithfulEnvelope)
        stored = store_typed_envelope(spec, tmp_path, _FaithfulEnvelope(_stub_payload()))
        assert stored.sha256 == compute_dataset_sha(_stub_payload())


class TestThePublicSurfaceExposesNoClassParameter:
    """A design pin, deliberately introspective, and worth its brittleness.

    The generic kernel underneath these functions takes the envelope class from
    a spec object. The whole safety argument rests on that spec never becoming
    a CALLER-supplied argument: a public ``cls=`` is exactly the hole that lets
    someone pass the wrong class and get a silently valid parse of the wrong
    type -- the same hole that kept ``ConditionSetEnvelope`` from inheriting
    from ``DatasetEnvelope`` in the first place.

    Pinned by signature rather than by behaviour because the failure is the
    EXISTENCE of the parameter, not any value of it; by the time a behavioural
    test could observe a wrong class being passed, the API has already shipped.
    """

    @pytest.mark.parametrize(
        ("store_fn", "load_fn"),
        [
            (store_dataset_envelope, load_dataset_envelope),
            (store_condition_set_envelope, load_condition_set_envelope),
        ],
        ids=["dataset", "condition_set"],
    )
    def test_the_public_functions_take_only_a_root_and_their_own_argument(
        self, store_fn: object, load_fn: object
    ) -> None:
        assert list(inspect.signature(store_fn).parameters) == ["root", "envelope"]  # type: ignore[arg-type]
        assert list(inspect.signature(load_fn).parameters) == ["root", "sha256"]  # type: ignore[arg-type]
