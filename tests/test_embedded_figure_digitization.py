"""Tests for :class:`carmel.schemas.datasets.EmbeddedFigureDigitization`.

Every fixture is SYNTHETIC; no paper text enters the repository.

The property under test is that a consumer holding ONLY an envelope's bytes can read how
complete a digitized series is, and cannot be handed an incoherent answer. That splits two ways:

- **Self-addressing** -- the embedded bytes are the canonical rendering of what they decode to,
  and they hash to the address they claim.
- **Reconstruction** -- the record they carry re-runs every construction invariant, so a
  ``coverage="complete"`` payload carrying an omission is refused on the bytes, with no
  producer in the loop and no document required. This is the check
  :class:`~carmel.schemas.datasets.EmbeddedTableInventory` documents itself as unable to make,
  and it is available here because a digitization record is a stated ledger rather than a
  derivation.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pickle
import sys
import textwrap
from typing import Any

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import DatasetEnvelope, EmbeddedFigureDigitization
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.figure_digitization_record import (
    DIGITIZATION_PAYLOAD_VERSION,
    UNREADABLE_PAYLOAD,
    CensusUnavailable,
    CensusUnavailableReason,
    FigureCoverage,
    FigureDigitization,
    MarkerCensus,
    MarkerOmission,
    MarkerOmissionReason,
    PlotRegion,
    digitization_record_bytes,
    digitization_record_payload,
)

RAW_SHA = "a" * 64
CROP_SHA = "b" * 64

REGION = PlotRegion(page=4, x_start=72.0, x_end=520.0, y_bottom=100.0, y_top=640.0)

STRADDLER = MarkerOmission(
    marker_id="m07",
    reason=MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE,
    x=519.4,
    y=302.1,
    detail="extent crosses the right axis",
)


def record(**overrides: Any) -> FigureDigitization:
    fields: dict[str, Any] = {
        "series_id": "fig3_phi_1",
        "raw_sha256": RAW_SHA,
        "figure_crop_node_id": "crop-fig3",
        "figure_crop_sha256": CROP_SHA,
        "plot_region": REGION,
        "coverage": FigureCoverage.PARTIAL,
        "census": MarkerCensus(detected=11),
        "recovered": 10,
        "omissions": (STRADDLER,),
    }
    fields.update(overrides)
    return FigureDigitization(**fields)


def embed(payload: dict[str, Any], **overrides: Any) -> EmbeddedFigureDigitization:
    """Embed a payload, computing a correct address and raw digest unless overridden."""
    canonical = digitization_record_bytes(payload)
    fields: dict[str, Any] = {
        "digitization_sha256": hashlib.sha256(canonical).hexdigest(),
        "raw_sha256": payload.get("raw_sha256", RAW_SHA),
        "canonical_json": canonical.decode("utf-8"),
    }
    fields.update(overrides)
    return EmbeddedFigureDigitization(**fields)


OVERFLOWING_HEX = "0x1p+99999"
"""A hex float naming a magnitude no ``float`` can hold.

``float.fromhex`` raises ``OverflowError`` on it -- NOT ``ValueError`` -- which is how it escaped
the documented refusal surface at every entry point until
:data:`~carmel.services.figure_digitization_record.UNREADABLE_PAYLOAD` grew the type.
"""


def _bytes_with_overflowing_coordinate() -> bytes:
    """Canonical bytes of an otherwise-valid payload whose x coordinate overflows on read."""
    payload = digitization_record_payload(record())
    payload["omissions"][0]["x"] = OVERFLOWING_HEX
    return digitization_record_bytes(payload)


class TestAConsumerHoldingOnlyTheEnvelopeBytes:
    """The two axes, read off an embedded citation and nothing else."""

    def test_a_partial_series_reads_as_partial_and_auditable(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        assert embedded.coverage is FigureCoverage.PARTIAL
        assert embedded.auditable is True
        assert embedded.omission_count == 1

    def test_a_complete_series_reads_as_complete(self) -> None:
        embedded = embed(
            digitization_record_payload(
                record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
            )
        )
        assert embedded.coverage is FigureCoverage.COMPLETE
        assert embedded.auditable is True
        assert embedded.omission_count == 0

    def test_an_unaudited_series_reads_as_uncheckable(self) -> None:
        embedded = embed(
            digitization_record_payload(
                record(
                    coverage=FigureCoverage.UNCHECKABLE,
                    census=CensusUnavailable(reason=CensusUnavailableReason.DETECTOR_UNAVAILABLE),
                    omissions=(),
                )
            )
        )
        assert embedded.coverage is FigureCoverage.UNCHECKABLE
        assert embedded.auditable is False
        assert embedded.omission_count == 0

    def test_whole_and_unaudited_never_read_alike(self) -> None:
        """Both have an empty ledger and the same recovered count; every axis separates them."""
        whole = embed(
            digitization_record_payload(
                record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
            )
        )
        unaudited = embed(
            digitization_record_payload(
                record(
                    coverage=FigureCoverage.UNCHECKABLE,
                    census=CensusUnavailable(reason=CensusUnavailableReason.SERIES_INSEPARABLE),
                    omissions=(),
                )
            )
        )
        assert whole.omission_count == unaudited.omission_count == 0
        assert whole.coverage is not unaudited.coverage
        assert whole.auditable is not unaudited.auditable

    def test_omissions_are_known_without_a_census(self) -> None:
        """UNCHECKABLE with a non-empty ledger: the two axes moving independently."""
        embedded = embed(
            digitization_record_payload(
                record(
                    coverage=FigureCoverage.UNCHECKABLE,
                    census=CensusUnavailable(reason=CensusUnavailableReason.ENUMERATION_INCOMPLETE),
                )
            )
        )
        assert embedded.coverage is FigureCoverage.UNCHECKABLE
        assert embedded.auditable is False
        assert embedded.omission_count == 1


class TestAnIncoherentRecordCannotBeCited:
    """The reconstruction, and what it forecloses."""

    def test_a_complete_claim_carrying_an_omission_is_refused_on_the_bytes(self) -> None:
        payload = digitization_record_payload(record())
        payload["coverage"] = FigureCoverage.COMPLETE.value
        with pytest.raises(ValidationError, match="does not reconstruct"):
            embed(payload)

    def test_a_census_that_does_not_balance_is_refused_on_the_bytes(self) -> None:
        payload = digitization_record_payload(
            record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
        )
        payload["census"] = {"detected": 13, "kind": "counted"}
        with pytest.raises(ValidationError, match="does not reconstruct"):
            embed(payload)

    def test_a_completeness_claim_without_a_census_is_refused_on_the_bytes(self) -> None:
        payload = digitization_record_payload(record())
        payload["census"] = {"detail": "", "kind": "unavailable", "reason": "detector_unavailable"}
        with pytest.raises(ValidationError, match="does not reconstruct"):
            embed(payload)

    def test_the_refusal_says_why_the_citation_is_worthless(self) -> None:
        payload = digitization_record_payload(record())
        payload["coverage"] = FigureCoverage.COMPLETE.value
        with pytest.raises(ValidationError, match="its coverage claim is not one anything could act on"):
            embed(payload)


class TestTheEmbeddedBytesAreSelfAddressing:
    """T1's five steps, each reachable on its own."""

    def test_a_well_formed_citation_is_accepted(self) -> None:
        payload = digitization_record_payload(record())
        embedded = embed(payload)
        assert embedded.raw_sha256 == RAW_SHA
        assert json.loads(embedded.canonical_json) == payload

    def test_canonical_json_that_does_not_parse_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="does not parse as JSON"):
            EmbeddedFigureDigitization(
                digitization_sha256=hashlib.sha256(b"{oops").hexdigest(),
                raw_sha256=RAW_SHA,
                canonical_json="{oops",
            )

    def test_canonical_json_that_is_not_an_object_is_refused(self) -> None:
        text = canonical_json_bytes([1, 2, 3]).decode("utf-8")
        with pytest.raises(ValidationError, match="is not a JSON object, it decodes to list"):
            EmbeddedFigureDigitization(
                digitization_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                raw_sha256=RAW_SHA,
                canonical_json=text,
            )

    def test_a_non_canonical_rendering_is_refused(self) -> None:
        payload = digitization_record_payload(record())
        pretty = json.dumps(payload, indent=2)
        with pytest.raises(ValidationError, match="is not the canonical rendering of what it decodes to"):
            EmbeddedFigureDigitization(
                digitization_sha256=hashlib.sha256(pretty.encode("utf-8")).hexdigest(),
                raw_sha256=RAW_SHA,
                canonical_json=pretty,
            )

    def test_a_record_at_the_wrong_address_is_refused(self) -> None:
        payload = digitization_record_payload(record())
        with pytest.raises(ValidationError, match="does not live at the address it claims"):
            embed(payload, digitization_sha256="c" * 64)

    def test_an_unreadable_version_is_named_as_such(self) -> None:
        payload = digitization_record_payload(record())
        payload["payload_version"] = DIGITIZATION_PAYLOAD_VERSION + 1
        with pytest.raises(ValidationError, match="is not the readable version"):
            embed(payload)

    def test_a_record_of_the_wrong_shape_is_refused(self) -> None:
        payload = digitization_record_payload(record())
        payload["confidence"] = "high"
        with pytest.raises(ValidationError, match=r"unexpected keys \['confidence'\]"):
            embed(payload)

    def test_a_record_naming_another_document_is_refused(self) -> None:
        payload = digitization_record_payload(record())
        with pytest.raises(ValidationError, match="the record names document"):
            embed(payload, raw_sha256="d" * 64)

    def test_a_malformed_digest_field_is_refused(self) -> None:
        payload = digitization_record_payload(record())
        for field in ("digitization_sha256", "raw_sha256"):
            with pytest.raises(ValidationError, match="is not 64 lowercase hex characters"):
                embed(payload, **{field: "A" * 64})

    def test_the_citation_is_frozen_and_forbids_extra_fields(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        with pytest.raises(ValidationError):
            embedded.digitization_sha256 = "e" * 64  # type: ignore[misc]
        with pytest.raises(ValidationError):
            EmbeddedFigureDigitization(
                digitization_sha256=embedded.digitization_sha256,
                raw_sha256=RAW_SHA,
                canonical_json=embedded.canonical_json,
                note="extra",  # type: ignore[call-arg]
            )


class TestTheReconstructionIsNotOptional:
    """Every read re-runs T1, so there is no such thing as an instance whose T1 did not run."""

    def test_the_accessors_answer_from_the_bytes_and_agree_across_reads(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        assert embedded.coverage is FigureCoverage.PARTIAL
        assert embedded.coverage is FigureCoverage.PARTIAL, "a second read must not derive a different answer"

    def test_installing_a_private_attribute_changes_nothing(self) -> None:
        """The strongest form of the guarantee: there is nothing private to attack.

        An earlier revision cached the reconstructed record, and writing that attribute made the
        object report the SWAPPED record's coverage. Now nothing reads it, so installing one --
        even a valid record saying something else entirely -- is inert. This asserts the absence
        of the attack surface rather than a defence against it.
        """
        embedded = embed(digitization_record_payload(record()))
        lying = record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
        for name in ("_record", "_record_for", "_field_digest"):
            object.__setattr__(embedded, name, lying)
        assert embedded.coverage is FigureCoverage.PARTIAL
        assert embedded.omission_count == 1


class TestTheAddressCannotSilentlyStopMatchingTheBytes:
    """`frozen=True` is a claim about attribute reassignment, and there are three ways around it.

    Each one used to leave this class reporting the ORIGINAL's coverage while carrying different
    bytes and an address that hashed neither. An addressed record whose address can drift from
    its bytes is worse than an unaddressed one: everything downstream that trusts the sha is then
    verifying a stale answer and reporting success.
    """

    def test_model_copy_with_an_update_revalidates(self) -> None:
        """The bytes go through T1 from the top, so the FIRST thing wrong with them is what
        fires -- here the canonical-rendering step, since ``"{}"`` is not canonical (``"{}\\n"``
        is). Which step fires is T1's business; that some step does is this test's."""
        embedded = embed(digitization_record_payload(record()))
        with pytest.raises(ValidationError, match="is not the canonical rendering of what it decodes to"):
            embedded.model_copy(update={"canonical_json": "{}"})

    def test_a_copy_carrying_canonical_bytes_still_fails_the_address_check(self) -> None:
        """Past the rendering step, the address is what refuses -- so the update path really is
        running all of T1 and not stopping at the first cheap check."""
        embedded = embed(digitization_record_payload(record()))
        with pytest.raises(ValidationError, match="does not live at the address it claims"):
            embedded.model_copy(update={"canonical_json": canonical_json_bytes({}).decode("utf-8")})

    def test_model_copy_will_not_mint_a_valid_citation_for_new_content(self) -> None:
        """It re-validates; it does NOT re-derive the address to match whatever arrived.

        Silently recomputing ``digitization_sha256`` would turn ``model_copy`` into a way to
        address arbitrary content, which is a worse hole than the one being closed.
        """
        embedded = embed(digitization_record_payload(record()))
        other = digitization_record_bytes(
            digitization_record_payload(record(series_id="fig9_other", census=MarkerCensus(detected=11)))
        ).decode("utf-8")
        with pytest.raises(ValidationError, match="does not live at the address it claims"):
            embedded.model_copy(update={"canonical_json": other})

    def test_a_coherent_update_of_bytes_and_address_together_is_accepted(self) -> None:
        """The guard refuses incoherence, not copying. Change both and it validates."""
        embedded = embed(digitization_record_payload(record()))
        payload = digitization_record_payload(
            record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
        )
        canonical = digitization_record_bytes(payload)
        updated = embedded.model_copy(
            update={
                "canonical_json": canonical.decode("utf-8"),
                "digitization_sha256": hashlib.sha256(canonical).hexdigest(),
            }
        )
        assert updated.coverage is FigureCoverage.COMPLETE
        assert updated.omission_count == 0

    def test_a_plain_copy_still_answers(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        assert embedded.model_copy().coverage is FigureCoverage.PARTIAL
        assert embedded.model_copy(deep=True).coverage is FigureCoverage.PARTIAL

    def test_model_construct_is_judged_on_its_bytes_like_everything_else(self) -> None:
        """`model_construct` skips construction validation, and the read validates regardless.

        The name was ``test_model_construct_answers_nothing``, and that was a universal the code
        refutes: an object built this way from a COHERENT triple answers, correctly, because the
        read re-derives from the bytes and those bytes are fine. Skipping the validator is not a
        crime the read punishes; it is simply not a way to be believed. Both directions are
        asserted here, since only asserting the refusal is what let the wrong name stand.

        This is also a deliberate behaviour change from the cached revision of this class, where
        a sentinel over the private cache refused EVERY ``model_construct`` object on sight
        (``self.__pydantic_private__`` being ``None`` was read as "this object skipped
        validation"). That sentinel was
        unsound anyway -- ``model_construct`` does not reliably leave ``None`` there, and under
        ``pytest --cov=carmel.schemas.datasets`` it left the ``ModelPrivateAttr`` DESCRIPTOR, so
        the check missed. There is no sentinel now because there is no cache. The coverage
        invocation stays in the verifier regardless.
        """
        incoherent = EmbeddedFigureDigitization.model_construct(
            digitization_sha256="c" * 64, raw_sha256=RAW_SHA, canonical_json="{}"
        )
        with pytest.raises(RuntimeError, match="no longer validates"):
            _ = incoherent.coverage

        honest = embed(digitization_record_payload(record()))
        coherent = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=honest.digitization_sha256,
            raw_sha256=honest.raw_sha256,
            canonical_json=honest.canonical_json,
        )
        assert coherent.coverage is FigureCoverage.PARTIAL
        assert coherent.omission_count == 1

    def test_writing_past_frozen_leaves_the_bytes_disowned(self) -> None:
        """`object.__setattr__` writes straight past `frozen=True` and cannot be prevented --
        so the accessors re-derive everything from the bytes rather than trusting anything."""
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "canonical_json", "{}")
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="no longer validates"):
                getattr(embedded, accessor)

    def test_the_refusal_carries_the_reason_t1_gave(self) -> None:
        """The wrapper must not swallow which of T1's refusals fired."""
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "canonical_json", "{}")
        with pytest.raises(RuntimeError, match="is not the canonical rendering of what it decodes to"):
            _ = embedded.coverage

    def test_a_stale_answer_is_never_returned_in_place_of_a_refusal(self) -> None:
        """The property all of the above exists for: no bypass yields the OLD coverage."""
        embedded = embed(
            digitization_record_payload(
                record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
            )
        )
        assert embedded.coverage is FigureCoverage.COMPLETE
        object.__setattr__(embedded, "canonical_json", "{}")
        with pytest.raises(RuntimeError):
            _ = embedded.coverage

    def test_rewriting_the_address_alone_is_caught(self) -> None:
        """The citation KEY, tampered.

        Provenance used to cover ``canonical_json`` only, so this left every accessor answering
        perfectly normally while ``digitization_sha256`` no longer hashed the payload -- the
        address-lying shape, one field over from where it was first closed.
        """
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "digitization_sha256", "0" * 64)
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="does not live at the address it claims"):
                getattr(embedded, accessor)

    def test_rewriting_the_document_digest_alone_is_caught(self) -> None:
        """``raw_sha256`` would then disagree with the payload's own, which T1 pins equal."""
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "raw_sha256", "d" * 64)
        with pytest.raises(RuntimeError, match="not the declared raw_sha256"):
            _ = embedded.coverage

    def test_no_single_field_tamper_yields_an_answer_instead_of_a_refusal(self) -> None:
        """Every SINGLE-attribute route, in one place.

        Renamed from a claim about "every route at once", which it never made: it tampers with
        one attribute at a time and never composes two, and the composed attacks are the ones
        that used to survive. Those live in
        :class:`TestTheGuardCannotBeRecomputedByWhoeverBypassedIt` below.
        """
        routes = (
            ("digitization_sha256", "0" * 64),
            ("raw_sha256", "d" * 64),
            ("canonical_json", "{}"),
        )
        for attribute, value in routes:
            embedded = embed(digitization_record_payload(record()))
            object.__setattr__(embedded, attribute, value)
            for accessor in ("coverage", "auditable", "omission_count"):
                with pytest.raises(RuntimeError):
                    getattr(embedded, accessor)


class TestTheGuardCannotBeRecomputedByWhoeverBypassedIt:
    """COMPOSED attacks: tamper, then repair whatever the guard was thought to consult.

    A single-field tamper is the easy case. The real attack is the second line, and an earlier
    revision fell to it: the read-time check compared a PRIVATE digest of the three PUBLIC
    fields, and every route that could write a public field past ``frozen=True`` could recompute
    that digest immediately afterwards. One extra statement and the object answered ``coverage``
    and ``auditable`` perfectly normally while its address named a hash of bytes it did not hold.

    The design that replaces it consults no stored provenance at all -- the address is re-derived
    from the bytes, and the document digest is read off the reconstructed record -- so there is
    nothing for a second line to repair. These tests are what hold that property down.
    """

    @staticmethod
    def _forgeable_state(embedded: EmbeddedFigureDigitization) -> dict[str, object]:
        """Everything an attacker can reach that is NOT one of the three declared fields.

        Three storage locations, because a memo can live in any of them and the guard is worth
        only as much as the narrowest one it checks: declared private attributes
        (``__pydantic_private__``), undeclared instance state (``__dict__`` beyond the fields),
        and pydantic's extras bag.
        """
        declared = set(EmbeddedFigureDigitization.model_fields)
        found: dict[str, object] = {}
        private = getattr(embedded, "__pydantic_private__", None)
        if private:
            found.update(private)
        found.update({k: v for k, v in vars(embedded).items() if k not in declared})
        extra = getattr(embedded, "__pydantic_extra__", None)
        if extra:
            found.update(extra)
        return found

    def test_the_class_holds_nothing_beyond_its_three_declared_fields(self) -> None:
        """The structural fix, asserted on BOTH axes an earlier version got wrong.

        Round one of this test named the two attributes it knew about, under a docstring
        promising that a future edit reintroducing "such an attribute" would fail here. It would
        not: an attribute called ``_field_digest`` passed both assertions. Round two fixed the
        NAMING axis with set equality and left the docstring promising "whatever it is called" --
        still false, now on the STORAGE axis, because ``_forgeable_state`` read
        ``__pydantic_private__`` alone. Memoizing into ``self.__dict__["_memo"]`` is the natural
        way to cache on a pydantic model without declaring a ``PrivateAttr``, and it passed.
        See :meth:`test_a_memo_hidden_outside_pydantics_private_bag_is_still_seen` for that
        counterexample, built rather than argued.

        Round three -- this one -- fixed a third axis: TIME. The assertion ran against an
        instance that had never been read, so a memo populated LAZILY inside
        :meth:`EmbeddedFigureDigitization._validated_record` passed it while the exploit was
        live. The object is now read before it is inspected, and read repeatedly, because "holds
        no state" is a claim about an instance in use rather than an instance just built.

        What this now promises, exactly: no state outside the three declared fields, in any of
        the three places INSTANCE state can live, before or after any number of reads. Two
        things it does not reach, named here rather than left for a later round to find:

        - Residue 3 of :meth:`EmbeddedFigureDigitization._validated_record` -- a subclass that
          overrides an accessor outright answers whatever it likes, and no assertion about
          storage can reach that.
        - A memo held OFF the instance, in a module-level or class-level map keyed by ``id`` or
          by ``digitization_sha256``. Nothing here would see it. What makes that unattractive
          rather than merely unguarded is that it is no cheaper than the memo this class already
          refuses to keep, and it inherits the same defect: an answer that outlives the bytes it
          was derived from.
        """
        embedded = embed(digitization_record_payload(record()))
        assert self._forgeable_state(embedded) == {}, "state before the first read"

        for _ in range(3):
            _ = embedded.coverage
            _ = embedded.auditable
            _ = embedded.omission_count
            assert self._forgeable_state(embedded) == {}, "a read left something behind"

    def test_a_memo_hidden_outside_pydantics_private_bag_is_still_seen(self) -> None:
        """The counterexample that made the previous docstring false, kept as the guard's proof.

        A subclass caching into ``__dict__`` reports ``COMPLETE`` over bytes that say ``PARTIAL``
        -- exactly the stale-answer bug this class was rewritten to end -- while the old
        ``__pydantic_private__``-only guard reported nothing to see.
        """

        class MemoizingSubclass(EmbeddedFigureDigitization):
            @property
            def coverage(self) -> FigureCoverage:
                memo = self.__dict__.get("_memo")
                if memo is None:
                    memo = self._reconstruct()
                    self.__dict__["_memo"] = memo
                return memo.coverage

        honest = embed(digitization_record_payload(record()))
        poisoned = MemoizingSubclass(
            digitization_sha256=honest.digitization_sha256,
            raw_sha256=honest.raw_sha256,
            canonical_json=honest.canonical_json,
        )
        assert poisoned.coverage is FigureCoverage.PARTIAL

        # A perfectly valid record that simply is not the one these bytes carry: the memo does
        # not even have to be malformed for the answer to be a lie.
        poisoned.__dict__["_memo"] = record(
            coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=()
        )
        assert poisoned.coverage is FigureCoverage.COMPLETE, "the exploit must actually work"

        # The old guard saw nothing. The current one sees the memo, which is the whole point.
        assert dict(getattr(poisoned, "__pydantic_private__", None) or {}) == {}
        assert "_memo" in self._forgeable_state(poisoned)

    def test_rewriting_the_address_and_repairing_every_private_attribute_still_refuses(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "digitization_sha256", "0" * 64)
        for name in self._forgeable_state(embedded):
            # Whatever private state exists, hand the attacker a chance to refresh it.
            object.__setattr__(embedded, name, getattr(embedded, name, None))
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="does not live at the address it claims"):
                getattr(embedded, accessor)

    def test_rewriting_the_document_digest_and_repairing_private_state_still_refuses(self) -> None:
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "raw_sha256", "d" * 64)
        for name in self._forgeable_state(embedded):
            object.__setattr__(embedded, name, getattr(embedded, name, None))
        with pytest.raises(RuntimeError, match="not the declared raw_sha256"):
            _ = embedded.coverage

    def test_swapping_the_bytes_and_the_cached_record_together_still_refuses(self) -> None:
        """The composed attack that a per-field check cannot see.

        Bytes and cache are replaced by a CONSISTENT pair -- the record really does serialize to
        the new bytes -- so any check pinning those two to each other passes. What is left
        untouched is ``digitization_sha256``, which still addresses the COMPLETE record, and a
        consumer that fetched by that address would be handed an object answering PARTIAL.
        """
        complete = record(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())
        embedded = embed(digitization_record_payload(complete))
        assert embedded.coverage is FigureCoverage.COMPLETE

        partial = record()
        object.__setattr__(
            embedded, "canonical_json", digitization_record_bytes(digitization_record_payload(partial)).decode("utf-8")
        )
        object.__setattr__(embedded, "_record", partial)
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="does not live at the address it claims"):
                getattr(embedded, accessor)

    def test_a_forged_object_does_not_survive_a_pickle_round_trip(self) -> None:
        """Serializing a forgery must not launder it into something that answers."""
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "digitization_sha256", "0" * 64)
        revived = pickle.loads(pickle.dumps(embedded))
        with pytest.raises(RuntimeError):
            _ = revived.coverage

    def test_an_honest_object_still_survives_a_pickle_round_trip(self) -> None:
        """The guards refuse forgeries, not ordinary use."""
        embedded = embed(digitization_record_payload(record()))
        revived = pickle.loads(pickle.dumps(embedded))
        assert revived.coverage is FigureCoverage.PARTIAL
        assert revived.auditable is True
        assert revived.omission_count == 1


class TestAReadRefusesEveryInvalidRecord:
    """The read path re-runs T1 entire, so no subset of it can admit an invalid record.

    ``FigureDigitization`` is a frozen DATACLASS, and ``object.__new__`` skips ``__post_init__``
    exactly as ``object.__setattr__`` skips pydantic's ``frozen=True``. A record minted that way
    can violate any of D1-D9. Pair it with bytes and an address generated to match, and every
    "is this object internally coherent" check passes -- address hashes the bytes, the record
    re-serializes to them, the document digests agree -- while
    :meth:`FigureDigitization.from_payload` refuses those identical bytes.

    Coherence is not validity. These tests are what hold the two together.

    The class was called ``TestAReadRefusesEverythingConstructionRefuses``, which is one word too
    wide: construction also applies this class's FIELD constraints, and a read does not re-apply
    them. ``test_construction_applies_a_size_bound_the_read_does_not`` is that exception. What is
    true without qualification is the name above -- no invalid RECORD survives a read.
    """

    @staticmethod
    def _unchecked_record(**overrides: Any) -> FigureDigitization:
        """A ``FigureDigitization`` whose ``__post_init__`` never ran."""
        built = object.__new__(FigureDigitization)
        fields: dict[str, Any] = {
            "series_id": "fig3_phi_1",
            "raw_sha256": RAW_SHA,
            "figure_crop_node_id": "crop-fig3",
            "figure_crop_sha256": CROP_SHA,
            "plot_region": REGION,
            "coverage": FigureCoverage.COMPLETE,
            "census": MarkerCensus(detected=11),
            "recovered": 10,
            "omissions": (STRADDLER,),
        }
        fields.update(overrides)
        for name, value in fields.items():
            object.__setattr__(built, name, value)
        return built

    @staticmethod
    def _install(canonical: bytes) -> EmbeddedFigureDigitization:
        """An ``EmbeddedFigureDigitization`` assembled without running its validator."""
        built = object.__new__(EmbeddedFigureDigitization)
        object.__setattr__(
            built,
            "__dict__",
            {
                "digitization_sha256": hashlib.sha256(canonical).hexdigest(),
                "raw_sha256": RAW_SHA,
                "canonical_json": canonical.decode("utf-8"),
            },
        )
        object.__setattr__(built, "__pydantic_private__", {})
        object.__setattr__(built, "__pydantic_extra__", None)
        object.__setattr__(built, "__pydantic_fields_set__", set())
        return built

    def test_the_bypass_really_does_produce_an_invalid_record(self) -> None:
        """The premise, checked -- otherwise the tests below prove nothing."""
        smuggled = self._unchecked_record()
        assert smuggled.coverage is FigureCoverage.COMPLETE
        assert smuggled.omissions, "the fixture must actually violate D8"
        with pytest.raises(ValueError, match="coverage=COMPLETE, but the record carries"):
            FigureDigitization.from_payload(digitization_record_payload(smuggled))

    def test_a_complete_claim_carrying_an_omission_is_refused_on_read(self) -> None:
        """D8, smuggled past the dataclass and caught by the read.

        Everything an internal-coherence check could ask for is true here: the address hashes
        the bytes, the record re-serializes to them, the document digests agree. Only re-running
        the record's own invariants catches it.
        """
        smuggled = self._unchecked_record()
        canonical = digitization_record_bytes(digitization_record_payload(smuggled))
        embedded = self._install(canonical)

        assert hashlib.sha256(canonical).hexdigest() == embedded.digitization_sha256
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="does not reconstruct"):
                getattr(embedded, accessor)

    def test_the_address_pins_the_rendering_not_the_spelling(self) -> None:
        """The limit of the canonical-rendering refusal, constructed rather than asserted away.

        Hex floats have two spellings per value, and ``canonical_json_bytes`` has no opinion
        about which one a string holds. So two payloads differing only in the CASE of one
        coordinate are each canonical JSON of themselves, each accepted at its own honest
        address, while ``from_payload`` returns equal records from both: one logical record at
        two addresses, which is what the refusal is justified as preventing.

        Not a live defect, because ``_pt`` emits ``float.hex()`` -- one spelling per value -- so
        no producer reaches the second address. That is the point of pinning it here: the axis
        is closed by the producer, and this fails if a future writer of coordinates stops
        closing it.
        """
        lower = digitization_record_payload(record())
        upper = digitization_record_payload(record())
        spelling = upper["omissions"][0]["x"]
        assert spelling != spelling.upper(), "the fixture needs a coordinate with two spellings"
        upper["omissions"][0]["x"] = spelling.upper()

        lower_bytes = digitization_record_bytes(lower)
        upper_bytes = digitization_record_bytes(upper)
        assert lower_bytes != upper_bytes
        assert hashlib.sha256(lower_bytes).hexdigest() != hashlib.sha256(upper_bytes).hexdigest()

        # Same record, two addresses, both accepted.
        assert FigureDigitization.from_payload(lower) == FigureDigitization.from_payload(upper)
        assert embed(lower).coverage is embed(upper).coverage is FigureCoverage.PARTIAL

        # And the producer is what keeps that unreachable.
        assert digitization_record_payload(record())["omissions"][0]["x"] == spelling

    def test_a_census_that_does_not_balance_is_refused_on_read(self) -> None:
        """D9, by the same route."""
        smuggled = self._unchecked_record(census=MarkerCensus(detected=11), recovered=10, omissions=())
        embedded = self._install(digitization_record_bytes(digitization_record_payload(smuggled)))
        with pytest.raises(RuntimeError, match="does not reconstruct"):
            _ = embedded.coverage

    def test_non_canonical_bytes_with_an_honest_address_are_refused_on_read(self) -> None:
        """The other half of why T1 is re-run WHOLE rather than as `from_payload` alone.

        These bytes are valid JSON of a valid record, and the address honestly hashes them -- so
        reconstruction alone would accept. What it would accept is one logical record with as
        many addresses as it has JSON renderings, which is addressing defeated.

        Renderings, precisely -- not spellings. See
        :meth:`test_the_address_pins_the_rendering_not_the_spelling` for the axis this check
        does not close and who does close it.
        """
        spaced = json.dumps(digitization_record_payload(record()), sort_keys=True, indent=1).encode("utf-8")
        embedded = self._install(spaced)

        assert hashlib.sha256(spaced).hexdigest() == embedded.digitization_sha256
        assert FigureDigitization.from_payload(json.loads(spaced)) == record()
        with pytest.raises(RuntimeError, match="is not the canonical rendering of what it decodes to"):
            _ = embedded.coverage

    def test_an_honestly_assembled_object_still_answers(self) -> None:
        """The same assembly path, with a VALID record: the guard refuses invalidity, not
        construction it did not personally witness."""
        embedded = self._install(digitization_record_bytes(digitization_record_payload(record())))
        assert embedded.coverage is FigureCoverage.PARTIAL
        assert embedded.omission_count == 1

    def test_construction_and_read_agree_on_whether_bytes_are_a_valid_record(self) -> None:
        """The invariant behind all of the above, asserted as the equivalence that HOLDS.

        The pairing here used to be ``from_payload`` against the accessor, under the name
        "...on every smuggled record", and that equivalence is false: ``from_payload`` is one of
        T1's refusals, not all of them.
        :meth:`test_non_canonical_bytes_with_an_honest_address_are_refused_on_read` in this class
        is a live counterexample -- ``from_payload`` accepts those bytes and the read refuses
        them, correctly. The old test passed only because its four hand-picked payloads all
        happened to be canonical.

        The property that does hold: on the question of whether these bytes are a valid record,
        canonically rendered, at the address they claim, construction and read are the SAME CALL
        and cannot diverge. The name is scoped to that question because the equivalence is not
        unconditional -- see
        :meth:`test_construction_applies_a_size_bound_the_read_does_not`, which is the exception,
        constructed rather than promised away.
        """
        cases: tuple[tuple[str, bytes], ...] = (
            ("valid", digitization_record_bytes(digitization_record_payload(record()))),
            ("d8 violated", digitization_record_bytes(digitization_record_payload(self._unchecked_record()))),
            (
                "d9 violated",
                digitization_record_bytes(
                    digitization_record_payload(
                        self._unchecked_record(census=MarkerCensus(detected=11), recovered=10, omissions=())
                    )
                ),
            ),
            (
                "d7 violated",
                digitization_record_bytes(
                    digitization_record_payload(
                        self._unchecked_record(coverage=FigureCoverage.PARTIAL, census=MarkerCensus(detected=11))
                    )
                ),
            ),
            # Valid record, honest address, non-canonical rendering: from_payload accepts these
            # and T1 does not. This is the case the old pairing could not have caught.
            ("non-canonical", json.dumps(digitization_record_payload(record()), sort_keys=True, indent=1).encode()),
            ("not an object", b"[]"),
            ("not json", b"{"),
            ("coordinate overflows a float", _bytes_with_overflowing_coordinate()),
        )
        for label, canonical in cases:
            address = hashlib.sha256(canonical).hexdigest()
            try:
                EmbeddedFigureDigitization(
                    digitization_sha256=address,
                    raw_sha256=RAW_SHA,
                    canonical_json=canonical.decode("utf-8"),
                )
            except ValidationError:
                construction_accepts = False
            else:
                construction_accepts = True

            embedded = self._install(canonical)
            try:
                _ = embedded.coverage
            except RuntimeError:
                read_accepts = False
            else:
                read_accepts = True

            assert construction_accepts == read_accepts, (
                f"construction and read disagree about {label}: "
                f"construction accepts={construction_accepts}, accessor accepts={read_accepts}"
            )

    def test_construction_applies_a_size_bound_the_read_does_not(self) -> None:
        """The one place construction and read part company, asserted so it stays known.

        ``canonical_json`` has a ``max_length``; the read re-derives the record and never
        re-checks the field constraints. So a VALID record whose canonical bytes exceed the
        bound is refused at construction and answered on read. Safe in the direction that
        matters -- the bound guards what may ENTER an envelope, and an object holding such bytes
        did not enter one -- but it is the reason this class's equivalence is scoped to validity
        rather than claimed for every triple of field values.
        """
        omissions = tuple(
            MarkerOmission(
                marker_id=f"m{index:06d}",
                reason=MarkerOmissionReason.OCCLUDED,
                x=200.0 + (index % 100) * 0.01,
                y=300.0,
                detail="d" * 40,
            )
            for index in range(9000)
        )
        oversized = record(census=MarkerCensus(detected=10 + len(omissions)), omissions=omissions)
        canonical = digitization_record_bytes(digitization_record_payload(oversized))
        assert len(canonical) > 1_048_576, "the fixture must actually exceed the embed bound"

        with pytest.raises(ValidationError, match="at most"):
            EmbeddedFigureDigitization(
                digitization_sha256=hashlib.sha256(canonical).hexdigest(),
                raw_sha256=RAW_SHA,
                canonical_json=canonical.decode("utf-8"),
            )
        assert self._install(canonical).coverage is FigureCoverage.PARTIAL

    def test_bytes_too_deeply_nested_to_re_serialize_are_refused_not_raised(self) -> None:
        """``json.loads`` uses a C scanner and ``canonical_json_bytes`` recurses in Python, so
        bytes can parse and then exhaust the stack one call later.

        At the default recursion limit the re-canonicalization survived every depth tried; this
        reproduces it by lowering the limit, which is the same situation a deep call chain
        creates. The guard exists because the stack left at that call belongs to the CALLER, not
        to the bytes.
        """
        depth = 400
        parsed = json.loads(("[" * depth) + ("]" * depth))
        nested = json.dumps({"payload_version": DIGITIZATION_PAYLOAD_VERSION, "x": parsed}).encode("utf-8")
        embedded = self._install(nested)

        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(200)
        try:
            with pytest.raises(RuntimeError, match="nested too deeply to re-serialize"):
                _ = embedded.coverage
        finally:
            sys.setrecursionlimit(limit)


class TestTheRefusalSurfaceHasNoUndocumentedEscapes:
    """A failing read raises ``RuntimeError``; a failing construction raises ``ValidationError``.

    That was false until this round, and false in a way that reading the code had not caught:
    making the read path re-run the whole reconstruction (``0861d52``) imported
    ``from_payload``'s refusal surface wholesale, and that surface leaked ``OverflowError`` out
    of ``float.fromhex``. A caller doing ``except RuntimeError`` around an accessor -- which is
    what the docstring tells them to do -- would have crashed on a stored coordinate.

    WHAT THE SWEEP COVERS, stated because "no undocumented escapes" is a claim about all inputs
    and this is a search, not a proof: one hostile VALUE substituted at a time, at every path of
    a well-formed payload, including interior containers. It therefore does not reach multi-field
    corruption, nesting depth (see
    :meth:`TestAReadRefusesEveryInvalidRecord.test_bytes_too_deeply_nested_to_re_serialize_are_refused_not_raised`),
    payload size, or an interpreter out of stack or memory. It found ``OverflowError`` at four
    entry points, which is the kind of thing it is for.
    """

    HOSTILE: tuple[Any, ...] = (
        None,
        True,
        0,
        -1,
        10**400,
        1.5,
        float("nan"),
        "",
        OVERFLOWING_HEX,
        "-" + OVERFLOWING_HEX,
        "0x1p-99999",
        "nan",
        "inf",
        "not a float",
        [],
        {},
        "\ud800",
        "\x00",
    )

    @staticmethod
    def _paths(node: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        """Every path into a payload: the ROOT, interior containers, and leaves.

        The empty path was excluded here (``[prefix] if prefix else []``), which meant every
        mutation was written INTO a parent container and the payload itself was never replaced.
        That is exactly the input ``from_payload``'s ``AttributeError`` needed, so the sweep
        could not have found the escape a commit message credited it with finding.
        """
        found = [prefix]
        if isinstance(node, dict):
            for key, value in node.items():
                found.extend(TestTheRefusalSurfaceHasNoUndocumentedEscapes._paths(value, (*prefix, key)))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                found.extend(TestTheRefusalSurfaceHasNoUndocumentedEscapes._paths(value, (*prefix, index)))
        return found

    @staticmethod
    def _mutated(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> Any:
        if not path:
            return value
        clone = json.loads(json.dumps(payload))
        node = clone
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value
        return clone

    @staticmethod
    def _render(payload: Any) -> bytes | None:
        """The bytes a store could hold for this payload, canonical where that is possible.

        Falls back to plain ``json.dumps`` for payloads ``canonical_json_bytes`` refuses -- those
        still reach a reader as bytes on disk, so they are exactly the inputs the read path has
        to refuse rather than crash on. Returns ``None`` only for payloads no JSON encoder can
        render, which no store could hold either.
        """
        try:
            return canonical_json_bytes(payload)
        except ValueError:
            pass
        try:
            return json.dumps(payload, sort_keys=True).encode("utf-8")
        except TypeError, ValueError:
            return None

    def test_no_mutation_escapes_the_documented_refusals(self) -> None:
        base = digitization_record_payload(record())
        paths = self._paths(base)
        assert () in paths, "the root must be substitutable, or the sweep cannot reach from_payload"

        checked = 0
        for path in paths:
            for value in self.HOSTILE:
                mutated = self._mutated(base, path, value)

                # from_payload FIRST, and with the raw mutated object rather than a re-decode:
                # it is a public entry point in its own right, and the only one a root
                # substitution reaches -- `_reconstruct` refuses a non-object before it ever
                # gets there, which is why sweeping the embedded class alone missed the
                # AttributeError this call now covers.
                try:
                    FigureDigitization.from_payload(mutated)
                except UNREADABLE_PAYLOAD:
                    pass
                except Exception as exc:  # noqa: BLE001 - the whole point is to catch the undocumented
                    pytest.fail(f"from_payload raised {type(exc).__name__} at {path} = {value!r}: {exc}")

                rendered = self._render(mutated)
                if rendered is None:
                    continue
                checked += 1
                address = hashlib.sha256(rendered).hexdigest()
                try:
                    EmbeddedFigureDigitization(
                        digitization_sha256=address,
                        raw_sha256=RAW_SHA,
                        canonical_json=rendered.decode("utf-8"),
                    )
                except ValidationError:
                    pass
                except Exception as exc:  # noqa: BLE001 - the whole point is to catch the undocumented
                    pytest.fail(f"construction raised {type(exc).__name__} at {path} = {value!r}: {exc}")

                embedded = TestAReadRefusesEveryInvalidRecord._install(rendered)
                for accessor in ("coverage", "auditable", "omission_count"):
                    try:
                        getattr(embedded, accessor)
                    except RuntimeError:
                        pass
                    except Exception as exc:  # noqa: BLE001
                        pytest.fail(f".{accessor} raised {type(exc).__name__} at {path} = {value!r}: {exc}")

        # Pinned exactly, not as a floor: a "> 300" bound let the shipped sweep be described as
        # 575 mutations when it ran 414, and nothing in the suite disagreed. If a path or a
        # hostile value is added, this number changes and says so.
        assert len(paths) == 24
        assert checked == 432, f"the sweep ran {checked} inputs, not the number this test claims"

    def test_the_overflowing_coordinate_really_does_overflow(self) -> None:
        """The premise behind the sweep's most interesting input, asserted not assumed."""
        with pytest.raises(OverflowError):
            float.fromhex(OVERFLOWING_HEX)

    def test_a_coordinate_too_large_for_a_float_is_refused_on_construction(self) -> None:
        canonical = _bytes_with_overflowing_coordinate()
        with pytest.raises(ValidationError, match="does not reconstruct"):
            EmbeddedFigureDigitization(
                digitization_sha256=hashlib.sha256(canonical).hexdigest(),
                raw_sha256=RAW_SHA,
                canonical_json=canonical.decode("utf-8"),
            )

    def test_a_coordinate_too_large_for_a_float_is_refused_on_read(self) -> None:
        """The exposure that is NEW in ``0861d52``: before the read re-ran the reconstruction,
        this input could not reach an accessor at all."""
        embedded = TestAReadRefusesEveryInvalidRecord._install(_bytes_with_overflowing_coordinate())
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="does not reconstruct"):
                getattr(embedded, accessor)


class TestTheDocstringSaysWhatIsActuallyTrue:
    """Claims that were false, pinned so they cannot drift back."""

    @staticmethod
    def _raised_messages() -> list[str]:
        """The message each ``raise`` in ``_reconstruct``'s BODY builds, docstring excluded.

        Read off the AST rather than off the source text, for two reasons that both bit the
        previous version of these tests. ``inspect.getsource`` includes the docstring, so
        ``phrase in source`` was the docstring quoting itself -- an assertion that could not
        fail, under a failure message describing a check it did not perform. And a text search
        for ``"raise ValueError("`` counts only refusals spelled exactly that way, so any other
        spelling was invisible to the count that turned out to be carrying the whole guard.

        An ``ast.Raise`` node's exception expression is walked whole, so an f-string's literal
        chunks are collected along with plain literals and joined in source order.
        """
        source = textwrap.dedent(inspect.getsource(EmbeddedFigureDigitization._reconstruct))
        function = ast.parse(source).body[0]
        assert isinstance(function, ast.FunctionDef)

        messages: list[str] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            literals = [
                part.value
                for part in ast.walk(node.exc)
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ]
            messages.append("".join(literals))
        return messages

    #: The marker phrases ``_reconstruct``'s docstring lists, in source order.
    REFUSALS = (
        "does not parse as JSON",
        "is not a JSON object",
        "nested too deeply to re-serialize",
        "is not the canonical rendering",
        "does not live at the address it claims",
        "is not the readable version",
        "is not the shape of a version-",
        "not the declared raw_sha256",
        "does not reconstruct",
    )

    def test_the_docstring_lists_a_refusal_the_code_actually_raises(self) -> None:
        """One listed phrase per raise, and one raise per listed phrase.

        A bijection rather than two containment checks, because containment is what let a
        code-only rename pass: renaming a message in the body while leaving the docstring alone
        kept the raise COUNT at nine and the docstring self-consistent, so the whole suite went
        green over a docstring listing a refusal the code no longer raised. Under a bijection
        that renamed raise matches no listed phrase and this fails.
        """
        doc = EmbeddedFigureDigitization._reconstruct.__doc__
        assert doc is not None
        messages = self._raised_messages()

        assert len(messages) == len(self.REFUSALS), (
            f"_reconstruct raises {len(messages)} times and its docstring lists "
            f"{len(self.REFUSALS)} refusals -- one of them is out of date"
        )

        for phrase in self.REFUSALS:
            assert f"``{phrase}``" in doc, f"{phrase!r} is listed nowhere in the docstring"
            matched = [message for message in messages if phrase in message]
            assert len(matched) == 1, (
                f"the docstring lists {phrase!r} and {len(matched)} of the raises in the BODY "
                f"build a message containing it"
            )

        for message in messages:
            matched_phrases = [phrase for phrase in self.REFUSALS if phrase in message]
            assert len(matched_phrases) == 1, (
                f"a raise building {message[:70]!r} matches {len(matched_phrases)} listed "
                f"refusals -- the docstring and the code have drifted apart"
            )

    def test_the_class_does_not_anchor_its_dormancy_to_series(self) -> None:
        """`Series` is a REQUIRED DatasetEnvelope field with min_length=1 and is projected into
        the envelope's identity payload -- the most-cited type in this module. Calling this class
        dormant "in exactly the way Series is" was false on the axis that was meant."""
        doc = EmbeddedFigureDigitization.__doc__
        assert doc is not None
        assert "DORMANT" not in doc
        assert "way :class:`Series`" not in doc
        assert "Nothing cites one of these today" in doc

    def test_series_really_is_a_required_envelope_field(self) -> None:
        """The fact that made the old sentence false, asserted rather than assumed."""
        field = DatasetEnvelope.model_fields["series"]
        assert field.is_required()

    def test_the_four_producer_checks_are_named(self) -> None:
        """They span the record and the Series, so nothing in this class can perform them --
        which is exactly why they have to be written down where a producer will look."""
        doc = EmbeddedFigureDigitization.__doc__
        assert doc is not None
        for check in (
            "record.series_id == series.series_id",
            "record.recovered == len(series.points)",
            "resolves to a :class:`SourceNode` of kind",
            "equals that node's ``sha256``",
        ):
            assert check in doc, f"the producer check {check!r} is documented nowhere"
