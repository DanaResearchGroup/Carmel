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

import hashlib
import json
import pickle
from typing import Any

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import DatasetEnvelope, EmbeddedFigureDigitization
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.figure_digitization_record import (
    DIGITIZATION_PAYLOAD_VERSION,
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

    def test_model_construct_answers_nothing(self) -> None:
        """`model_construct` skips construction validation, and the read runs it anyway.

        Previously this depended on a sentinel over the private cache, which was unsound:
        ``model_construct`` does not reliably leave ``None`` there, and under
        ``pytest --cov=carmel.schemas.datasets`` it left the ``ModelPrivateAttr`` DESCRIPTOR, so
        the check missed. There is no sentinel now because there is no cache -- the read simply
        validates. The coverage invocation stays in the verifier regardless.
        """
        built = EmbeddedFigureDigitization.model_construct(
            digitization_sha256="c" * 64, raw_sha256=RAW_SHA, canonical_json="{}"
        )
        with pytest.raises(RuntimeError, match="no longer validates"):
            _ = built.coverage

    def test_writing_past_frozen_leaves_the_bytes_disowned(self) -> None:
        """`object.__setattr__` writes straight past `frozen=True` and cannot be prevented --
        so the accessors re-derive everything from the bytes rather than trusting anything."""
        embedded = embed(digitization_record_payload(record()))
        object.__setattr__(embedded, "canonical_json", "{}")
        for accessor in ("coverage", "auditable", "omission_count"):
            with pytest.raises(RuntimeError, match="no longer validates"):
                getattr(embedded, accessor)

    def test_the_refusal_carries_the_reason_t1_gave(self) -> None:
        """The wrapper must not swallow which of T1's six steps fired."""
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
        """Everything private an attacker can reach, so a test can try to repair all of it."""
        private = getattr(embedded, "__pydantic_private__", None)
        return dict(private) if private else {}

    def test_the_class_holds_no_private_state_at_all(self) -> None:
        """The structural fix, asserted NAME-AGNOSTICALLY.

        An earlier version of this test named the two attributes it knew about, under a docstring
        promising that a future edit reintroducing "such an attribute" would fail here. It would
        not: an attribute called ``_field_digest`` passed both assertions. A guard whose claim is
        broader than its construction is the exact defect this whole ticket is about, and it had
        got into the test meant to catch it.

        Set equality is what makes the promise true. It is empty because
        :class:`EmbeddedFigureDigitization` now derives everything on read; if a cache is ever
        reintroduced, whatever it is called, this fails before the exploit is written.
        """
        embedded = embed(digitization_record_payload(record()))
        assert self._forgeable_state(embedded) == {}

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


class TestAReadRefusesEverythingConstructionRefuses:
    """The read path re-runs T1 entire, so no subset of it can admit an invalid record.

    ``FigureDigitization`` is a frozen DATACLASS, and ``object.__new__`` skips ``__post_init__``
    exactly as ``object.__setattr__`` skips pydantic's ``frozen=True``. A record minted that way
    can violate any of D1-D9. Pair it with bytes and an address generated to match, and every
    "is this object internally coherent" check passes -- address hashes the bytes, the record
    re-serializes to them, the document digests agree -- while
    :meth:`FigureDigitization.from_payload` refuses those identical bytes.

    Coherence is not validity. These tests are what hold the two together.
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
        many addresses as it has renderings, which is addressing defeated.
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

    def test_construction_and_read_agree_on_every_smuggled_record(self) -> None:
        """The invariant behind all of the above, asserted as an equivalence.

        For each payload: whatever ``from_payload`` does, the accessor does. A read that accepted
        what construction refuses is the bug this class had; a read that refused what
        construction accepts would be a new one.
        """
        candidates = (
            self._unchecked_record(),
            self._unchecked_record(census=MarkerCensus(detected=11), recovered=10, omissions=()),
            self._unchecked_record(coverage=FigureCoverage.PARTIAL, census=MarkerCensus(detected=11)),
            record(),
        )
        for candidate in candidates:
            canonical = digitization_record_bytes(digitization_record_payload(candidate))
            try:
                FigureDigitization.from_payload(json.loads(canonical))
            except ValueError:
                constructor_accepts = False
            else:
                constructor_accepts = True

            embedded = self._install(canonical)
            try:
                _ = embedded.coverage
            except RuntimeError:
                read_accepts = False
            else:
                read_accepts = True

            assert constructor_accepts == read_accepts, (
                f"construction and read disagree about {candidate.series_id!r}: "
                f"from_payload accepts={constructor_accepts}, accessor accepts={read_accepts}"
            )


class TestTheDocstringSaysWhatIsActuallyTrue:
    """Two claims that were false, pinned so they cannot drift back."""

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
