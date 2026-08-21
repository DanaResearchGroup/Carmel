"""Tests for the figure digitization record.

Every fixture is SYNTHETIC; no paper text enters the repository.

The property under test is that PARTIALNESS is something the stored bytes STATE, and state
coherently. Two things follow from that and are tested separately throughout, because fusing
them is the failure the record was written to end:

- **Coverage** -- is anything missing?
- **Auditability** -- could the instrument have told?

The three readings a holder of the record gets, COMPLETE / PARTIAL / UNCHECKABLE, are pinned
first (see ``TestAReaderHoldingOnlyTheRecord``) and are pinned against the PAYLOAD DICT, never
against a live ``FigureDigitization`` -- if telling them apart needed anything the bytes do not
carry, the record would be wrong and the test would be hiding it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from carmel.services import figure_digitization_record
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.figure_digitization_record import (
    DIGITIZATION_PAYLOAD_KEYS,
    DIGITIZATION_PAYLOAD_VERSION,
    CensusUnavailable,
    CensusUnavailableReason,
    FigureCoverage,
    FigureDigitization,
    MarkerCensus,
    MarkerOmission,
    MarkerOmissionReason,
    PlotRegion,
    census_of,
    compute_digitization_sha,
    coverage_of,
    digitization_record_bytes,
    digitization_record_payload,
    is_auditable,
    omission_reasons_of,
    payload_unreadable_reason,
)

RAW_SHA = "a" * 64
CROP_SHA = "b" * 64

REGION = PlotRegion(page=4, x_start=72.0, x_end=520.0, y_bottom=100.0, y_top=640.0)

#: A marker sitting on the right-hand axis boundary (x=520.0). It stays EXCLUDED -- the record's
#: job is to say it was seen and dropped, never to rescue it by choosing a coordinate the figure
#: does not determine.
STRADDLER = MarkerOmission(
    marker_id="m07",
    reason=MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE,
    x=519.4,
    y=302.1,
    detail="centre 0.6 pt inside the right axis, extent crosses it",
)

OCCLUDED = MarkerOmission(
    marker_id="m11",
    reason=MarkerOmissionReason.OCCLUDED,
    x=310.0,
    y=288.5,
    detail="overlapped where the phi=1.0 curve crosses",
)


def digitization(**overrides: Any) -> FigureDigitization:
    """A PARTIAL record: 12 markers counted, 10 recovered, 2 itemised. The balanced default."""
    fields: dict[str, Any] = {
        "series_id": "fig3_phi_1",
        "raw_sha256": RAW_SHA,
        "figure_crop_node_id": "crop-fig3",
        "figure_crop_sha256": CROP_SHA,
        "plot_region": REGION,
        "coverage": FigureCoverage.PARTIAL,
        "census": MarkerCensus(detected=12),
        "recovered": 10,
        "omissions": (STRADDLER, OCCLUDED),
    }
    fields.update(overrides)
    return FigureDigitization(**fields)


def complete() -> FigureDigitization:
    """The same series with nothing lost: a census of 10, 10 recovered, an empty ledger."""
    return digitization(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=10), omissions=())


def uncheckable(**overrides: Any) -> FigureDigitization:
    """The same series with no census: the detector never ran, so no total exists."""
    fields: dict[str, Any] = {
        "coverage": FigureCoverage.UNCHECKABLE,
        "census": CensusUnavailable(
            reason=CensusUnavailableReason.DETECTOR_UNAVAILABLE,
            detail="digitized by hand off a printed copy",
        ),
        "omissions": (),
    }
    fields.update(overrides)
    return digitization(**fields)


class TestAReaderHoldingOnlyTheRecord:
    """The three states, told apart from the payload dict alone.

    Every assertion here goes through a ``json.loads(json.dumps(...))`` round trip first, so
    nothing under test can be answered from a Python object that survived in memory. What the
    reader holds is what a file holds.
    """

    @staticmethod
    def _as_a_reader_would_hold_it(record: FigureDigitization) -> dict[str, Any]:
        return json.loads(digitization_record_bytes(digitization_record_payload(record)).decode("utf-8"))  # type: ignore[no-any-return]

    def test_complete_is_nothing_missing_and_the_instrument_could_tell(self) -> None:
        payload = self._as_a_reader_would_hold_it(complete())
        assert coverage_of(payload) is FigureCoverage.COMPLETE
        assert is_auditable(payload) is True
        assert omission_reasons_of(payload) == ()

    def test_partial_is_something_missing_and_itemised(self) -> None:
        payload = self._as_a_reader_would_hold_it(digitization())
        assert coverage_of(payload) is FigureCoverage.PARTIAL
        assert is_auditable(payload) is True
        assert omission_reasons_of(payload) == (
            MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE,
            MarkerOmissionReason.OCCLUDED,
        )

    def test_uncheckable_is_no_way_to_know(self) -> None:
        payload = self._as_a_reader_would_hold_it(uncheckable())
        assert coverage_of(payload) is FigureCoverage.UNCHECKABLE
        assert is_auditable(payload) is False
        assert census_of(payload) == CensusUnavailable(
            reason=CensusUnavailableReason.DETECTOR_UNAVAILABLE,
            detail="digitized by hand off a printed copy",
        )

    def test_complete_and_uncheckable_do_not_collapse(self) -> None:
        """The pair this record exists for.

        Both series have an EMPTY omission ledger. One is whole and the other is unaudited, and
        before this record they rendered identically -- a point count and nothing else. Here
        every axis that could tell them apart does.
        """
        whole = self._as_a_reader_would_hold_it(complete())
        unaudited = self._as_a_reader_would_hold_it(uncheckable())

        assert omission_reasons_of(whole) == omission_reasons_of(unaudited) == ()
        assert whole["recovered"] == unaudited["recovered"]

        assert coverage_of(whole) is not coverage_of(unaudited)
        assert is_auditable(whole) is not is_auditable(unaudited)
        assert whole["census"] != unaudited["census"]

    def test_the_two_axes_are_independent_not_one_field_twice(self) -> None:
        """UNCHECKABLE with a NON-EMPTY ledger: known omissions, and no bound on them.

        This is the cell that proves the axes are orthogonal rather than two spellings of one
        fact. The run named an omission and then lost the ability to enumerate, so it knows one
        marker is missing and cannot say only one is. Reporting it as PARTIAL would assert a
        bound nothing established; reporting the ledger as empty would lose what it does know.
        """
        payload = self._as_a_reader_would_hold_it(
            digitization(
                coverage=FigureCoverage.UNCHECKABLE,
                census=CensusUnavailable(reason=CensusUnavailableReason.ENUMERATION_INCOMPLETE),
                omissions=(STRADDLER,),
            )
        )
        assert coverage_of(payload) is FigureCoverage.UNCHECKABLE
        assert is_auditable(payload) is False
        assert omission_reasons_of(payload) == (MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE,)

    def test_an_unreadable_payload_never_answers_that_the_series_is_fine(self) -> None:
        """Every reader raises rather than defaulting. A default here reads as reassurance."""
        for reader in (coverage_of, census_of, is_auditable, omission_reasons_of):
            with pytest.raises(ValueError):
                reader({"payload_version": 99})


class TestACompleteClaimCannotCarryAnOmission:
    """The illegal state, made unconstructible rather than documented."""

    def test_construction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="coverage=COMPLETE, but the record carries 2 omission"):
            digitization(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=12))

    def test_the_refusal_names_the_markers(self) -> None:
        with pytest.raises(ValueError, match=r"\['m07', 'm11'\]"):
            digitization(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=12))

    def test_it_cannot_be_smuggled_in_through_a_stored_payload(self) -> None:
        """The same refusal on bytes that arrived from outside the process.

        A record built in memory and one parsed off disk are held to the SAME invariants --
        otherwise the guard is one a producer can step around by writing the file itself.
        """
        payload = digitization_record_payload(digitization())
        payload["coverage"] = FigureCoverage.COMPLETE.value
        with pytest.raises(ValueError, match="coverage=COMPLETE, but the record carries 2 omission"):
            FigureDigitization.from_payload(payload)

    def test_a_partial_claim_with_an_empty_ledger_is_refused_too(self) -> None:
        """Partialness a reader cannot itemise is a warning, not a record."""
        with pytest.raises(ValueError, match="coverage=PARTIAL, but the omission ledger is empty"):
            digitization(census=MarkerCensus(detected=10), omissions=())


class TestTheCensusMustBalance:
    """What stops a COMPLETE claim from being true only of the fields its own producer wrote."""

    def test_markers_that_vanished_unrecorded_are_refused(self) -> None:
        """12 counted, 9 recovered, nothing in the ledger: three markers lost silently.

        Without this check the record would pass every coverage test above and claim COMPLETE --
        exactly the failure the ticket describes, a series that lost markers reading as whole.
        """
        with pytest.raises(ValueError, match="the census does not balance -- detected=12"):
            digitization(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=12), omissions=(), recovered=9)

    def test_a_ledger_longer_than_the_census_is_refused(self) -> None:
        with pytest.raises(ValueError, match="the census does not balance"):
            digitization(census=MarkerCensus(detected=11))

    def test_the_balanced_record_is_accepted(self) -> None:
        record = digitization()
        assert record.census == MarkerCensus(detected=12)
        assert record.recovered + len(record.omissions) == 12

    def test_no_balance_is_struck_when_there_is_no_total(self) -> None:
        """With no census there is no arithmetic to do, and none is invented."""
        record = uncheckable(recovered=3, omissions=(STRADDLER, OCCLUDED))
        assert record.recovered == 3
        assert len(record.omissions) == 2


class TestCoverageAndAuditabilityArePinnedToEachOther:
    """D7, in both directions."""

    def test_a_completeness_claim_without_a_census_is_refused(self) -> None:
        for claim in (FigureCoverage.COMPLETE, FigureCoverage.PARTIAL):
            with pytest.raises(ValueError, match="the only honest coverage here is UNCHECKABLE"):
                digitization(
                    coverage=claim,
                    census=CensusUnavailable(reason=CensusUnavailableReason.PLOT_REGION_UNBOUNDED),
                )

    def test_a_record_may_not_decline_a_question_its_own_evidence_settles(self) -> None:
        with pytest.raises(ValueError, match="coverage=UNCHECKABLE, but this record carries a census"):
            digitization(coverage=FigureCoverage.UNCHECKABLE)

    def test_every_census_unavailable_reason_reaches_uncheckable(self) -> None:
        for reason in CensusUnavailableReason:
            record = uncheckable(census=CensusUnavailable(reason=reason))
            assert record.coverage is FigureCoverage.UNCHECKABLE


class TestTheStoredFormIsCanonicalAndSelfAddressing:
    """The bytes, their address, and the trip through a real file."""

    def test_canonical_bytes_hash_to_the_records_own_address(self) -> None:
        payload = digitization_record_payload(digitization())
        assert compute_digitization_sha(payload) == hashlib.sha256(digitization_record_bytes(payload)).hexdigest()

    def test_the_address_is_the_table_lanes_rule_not_a_second_one(self) -> None:
        """Addressed by sha256 over ``canonical_json_bytes``, exactly as the table lane is."""
        payload = digitization_record_payload(digitization())
        assert digitization_record_bytes(payload) == canonical_json_bytes(dict(payload))
        assert compute_digitization_sha(payload) == hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()

    def test_the_address_is_independent_of_key_order(self) -> None:
        payload = digitization_record_payload(digitization())
        shuffled = dict(reversed(list(payload.items())))
        assert compute_digitization_sha(shuffled) == compute_digitization_sha(payload)

    def test_a_stored_record_round_trips_through_a_file(self, tmp_path: Path) -> None:
        """Record -> payload -> canonical bytes -> disk -> parse -> record, all identical.

        Written to a real file under its own address, and read back through ``json.loads``, so
        nothing in the chain can be satisfied by an object that never left the process.
        """
        record = digitization()
        payload = digitization_record_payload(record)
        address = compute_digitization_sha(payload)

        path = tmp_path / f"{address}.json"
        path.write_bytes(digitization_record_bytes(payload))

        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == address, "the file does not hash to the name it sits under"

        reloaded = json.loads(raw.decode("utf-8"))
        assert digitization_record_bytes(reloaded) == raw, "what came back is not in canonical form"
        assert FigureDigitization.from_payload(reloaded) == record
        assert compute_digitization_sha(reloaded) == address

    def test_the_round_trip_preserves_both_axes_for_all_three_states(self, tmp_path: Path) -> None:
        for name, record in (("complete", complete()), ("partial", digitization()), ("uncheckable", uncheckable())):
            payload = digitization_record_payload(record)
            path = tmp_path / f"{name}.json"
            path.write_bytes(digitization_record_bytes(payload))
            reloaded = json.loads(path.read_bytes().decode("utf-8"))
            assert coverage_of(reloaded) is record.coverage
            assert is_auditable(reloaded) is isinstance(record.census, MarkerCensus)

    def test_geometry_survives_the_trip_bit_for_bit(self) -> None:
        """``float.hex`` round trips exactly; a decimal rendering would not.

        The awkward values are built to land INSIDE the plot region, because D6b refuses an
        omission centred outside it, and each is a sum that binary floating point cannot
        represent exactly -- ``200.1 + 0.2`` is ``200.29999999999998``. Both facts are asserted,
        so a later edit cannot quietly swap in a value that round trips because it is exact.
        """
        awkward = replace(STRADDLER, x=200.1 + 0.2, y=128.1 + 0.2)
        assert awkward.x != 200.3 and awkward.y != 128.3, "the fixture no longer exercises inexact floats"

        record = digitization(omissions=(awkward, OCCLUDED))
        rebuilt = FigureDigitization.from_payload(digitization_record_payload(record))
        assert rebuilt.omissions[0].x == 200.1 + 0.2
        assert rebuilt.omissions[0].y == 128.1 + 0.2

    def test_a_non_finite_coordinate_never_inhabits_a_record(self) -> None:
        """Refused at CONSTRUCTION of the type that holds it, not at serialization.

        ``float('nan').hex()`` is the valid string ``'nan'``, so a record built with one would
        otherwise round-trip into the store looking like a measurement -- and by the time
        serialization noticed, the record would already have been compared and passed around.

        The refusal must also name the COORDINATE. ``PlotRegion``'s ordering checks refuse a
        ``nan`` on their own, because every comparison against one is False, but they refuse it
        as "spans no height" -- which sends a reader looking for a box that was never the
        problem. Asserting on the message is what holds that apart.
        """
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match=r"MarkerOmission\('m07'\).x: refusing to record a non-finite"):
                replace(STRADDLER, x=bad)
            with pytest.raises(ValueError, match="PlotRegion.y_top: refusing to record a non-finite"):
                replace(REGION, y_top=bad)


class TestTheDeclaredShapeIsTheShapeActuallyWritten:
    """The key set and the version, pinned against a real built payload."""

    def test_the_payload_keys_constant_matches_a_built_payload(self) -> None:
        assert set(digitization_record_payload(digitization())) == set(DIGITIZATION_PAYLOAD_KEYS)

    def test_the_payload_declares_the_version_this_code_reads(self) -> None:
        assert digitization_record_payload(digitization())["payload_version"] == DIGITIZATION_PAYLOAD_VERSION

    def test_an_unknown_version_is_unreadable_not_incoherent(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["payload_version"] = DIGITIZATION_PAYLOAD_VERSION + 1
        with pytest.raises(ValueError, match="is not the readable version"):
            FigureDigitization.from_payload(payload)

    def test_a_stray_key_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["confidence"] = "high"
        with pytest.raises(ValueError, match=r"unexpected keys \['confidence'\]"):
            FigureDigitization.from_payload(payload)

    def test_a_missing_key_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        del payload["census"]
        with pytest.raises(ValueError, match=r"missing keys \['census'\]"):
            FigureDigitization.from_payload(payload)


class TestReadingAnUntrustedPayload:
    """``from_payload`` is handed bytes from a file, so every field is hostile input."""

    def test_an_untagged_census_is_refused_rather_than_guessed(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["census"] = {"detected": 12}
        with pytest.raises(ValueError, match="an untagged census cannot be read"):
            FigureDigitization.from_payload(payload)

    def test_a_census_that_is_not_an_object_states_no_auditability(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["census"] = 12
        with pytest.raises(ValueError, match="'census' is int, not an object"):
            FigureDigitization.from_payload(payload)

    def test_a_boolean_count_is_not_a_count(self) -> None:
        """``isinstance(True, int)``, so ``recovered: true`` would arithmetic as 1."""
        payload = digitization_record_payload(digitization())
        payload["recovered"] = True
        with pytest.raises(ValueError, match="recovered: a count must be an int, got bool"):
            FigureDigitization.from_payload(payload)

    def test_a_stringified_field_is_not_coerced_into_looking_valid(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["series_id"] = None
        with pytest.raises(ValueError, match="series_id is NoneType, not a string"):
            FigureDigitization.from_payload(payload)

    def test_an_omissions_list_that_is_not_a_list_itemises_nothing(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["omissions"] = {"m07": "straddle"}
        with pytest.raises(ValueError, match="'omissions' is dict, not a list"):
            FigureDigitization.from_payload(payload)

    def test_an_omission_that_is_not_an_object_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["omissions"] = ["m07", "m11"]
        with pytest.raises(ValueError, match=r"omissions\[0\] is str, not an object"):
            FigureDigitization.from_payload(payload)

    def test_a_plot_region_that_is_not_an_object_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["plot_region"] = "72,520,100,640"
        with pytest.raises(ValueError, match="'plot_region' is str, not an object"):
            FigureDigitization.from_payload(payload)

    def test_an_unknown_omission_reason_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["omissions"][0]["reason"] = "we_did_not_like_it"
        with pytest.raises(ValueError, match="we_did_not_like_it"):
            FigureDigitization.from_payload(payload)


class TestPayloadUnreadableReason:
    """The pre-flight a citing layer runs before a record is embedded."""

    def test_a_readable_record_reports_none(self) -> None:
        assert payload_unreadable_reason(digitization_record_payload(digitization())) is None

    def test_it_performs_the_same_read_the_readers_perform(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["coverage"] = FigureCoverage.COMPLETE.value
        reason = payload_unreadable_reason(payload)
        assert reason is not None
        assert "coverage=COMPLETE" in reason

    def test_it_catches_a_key_error_rather_than_letting_it_escape(self) -> None:
        """``from_payload`` indexes into an untrusted mapping; a bare ``KeyError`` out of this
        function would crash a caller that correctly catches only ``ValueError``."""
        payload = digitization_record_payload(digitization())
        del payload["plot_region"]["y_top"]
        reason = payload_unreadable_reason(payload)
        assert reason is not None
        assert "y_top" in reason

    def test_it_catches_a_type_error_rather_than_letting_it_escape(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["plot_region"]["y_top"] = None
        assert payload_unreadable_reason(payload) is not None

    def test_a_payload_of_the_wrong_shape_entirely_reports_a_reason(self) -> None:
        assert payload_unreadable_reason({}) is not None


class TestIdentityAndLedgerHygiene:
    """D1-D6: the fields that make a record joinable, and the ledger itemised."""

    def test_a_series_id_that_is_not_an_identifier_is_refused(self) -> None:
        for bad in ("Fig3", "3fig", "fig 3", "", "fig-3"):
            with pytest.raises(ValueError, match="is not a lowercase identifier"):
                digitization(series_id=bad)

    def test_a_malformed_digest_is_refused_on_both_digest_fields(self) -> None:
        for field in ("raw_sha256", "figure_crop_sha256"):
            with pytest.raises(ValueError, match=f"{field} must be 64 lowercase hex"):
                digitization(**{field: "A" * 64})
            with pytest.raises(ValueError, match=f"{field} must be 64 lowercase hex"):
                digitization(**{field: "a" * 63})

    def test_a_crop_node_id_must_be_a_usable_handle(self) -> None:
        for bad in ("", "  ", " crop-fig3"):
            with pytest.raises(ValueError, match="figure_crop_node_id must be non-empty"):
                digitization(figure_crop_node_id=bad)

    def test_both_halves_of_the_crops_node_identity_are_carried(self) -> None:
        """``node_id`` alone is unique only within one envelope's graph; ``sha256`` alone cannot
        say which node was meant when two hold identical bytes."""
        payload = digitization_record_payload(digitization())
        assert payload["figure_crop_node_id"] == "crop-fig3"
        assert payload["figure_crop_sha256"] == CROP_SHA

    def test_a_record_that_recovered_nothing_is_not_a_record_about_a_series(self) -> None:
        with pytest.raises(ValueError, match="a digitization that recovered nothing is a refusal"):
            digitization(coverage=FigureCoverage.COMPLETE, census=MarkerCensus(detected=0), omissions=(), recovered=0)

    def test_a_repeated_marker_id_is_refused(self) -> None:
        twin = replace(OCCLUDED, marker_id="m07")
        with pytest.raises(ValueError, match="duplicate marker_id 'm07'"):
            digitization(omissions=(STRADDLER, twin))

    def test_an_unsorted_ledger_is_refused(self) -> None:
        with pytest.raises(ValueError, match="omissions must be sorted ascending by marker_id"):
            digitization(omissions=(OCCLUDED, STRADDLER))

    def test_an_empty_marker_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="marker_id must be non-empty"):
            MarkerOmission(marker_id="", reason=MarkerOmissionReason.OCCLUDED, x=1.0, y=2.0)

    def test_a_negative_census_is_refused(self) -> None:
        with pytest.raises(ValueError, match="MarkerCensus.detected: a count cannot be negative"):
            MarkerCensus(detected=-1)


class TestThePlotRegionGivesTheLedgerAReferent:
    """Without a frame, an omission's coordinates are uninterpretable numbers."""

    def test_a_degenerate_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="PlotRegion spans no width"):
            replace(REGION, x_end=72.0)
        with pytest.raises(ValueError, match="PlotRegion spans no height"):
            replace(REGION, y_top=100.0)

    def test_an_inverted_frame_is_refused(self) -> None:
        with pytest.raises(ValueError, match="PlotRegion spans no width"):
            replace(REGION, x_start=600.0)

    def test_a_page_ordinal_must_be_one_based(self) -> None:
        with pytest.raises(ValueError, match="must be a 1-based page number"):
            replace(REGION, page=0)

    def test_the_straddler_sits_on_the_boundary_the_frame_declares(self) -> None:
        """The ledger entry is checkable against the frame, which is why the frame is stored."""
        record = digitization()
        assert record.omissions[0].reason is MarkerOmissionReason.AXIS_BOUNDARY_STRADDLE
        assert record.omissions[0].x < record.plot_region.x_end
        assert record.plot_region.x_end - record.omissions[0].x < 1.0


class TestWhatCanBeBuiltIsWhatCanBeReadBack:
    """Type hints are not guards, and a record built past one reaches the store broken.

    Every value below used to CONSTRUCT and then fail later -- either refused by this module's
    own ``from_payload`` after being serialized as a JSON boolean, or crashing inside ``_pt``,
    which has no ``.hex()`` to call on an int. The failure landed at the moment of WRITING rather
    than the moment of the mistake, which is the worst place for it.
    """

    def test_a_bool_is_refused_wherever_a_count_is_meant(self) -> None:
        with pytest.raises(ValueError, match="MarkerCensus.detected: a count must be an int, got bool"):
            MarkerCensus(detected=True)
        with pytest.raises(ValueError, match="PlotRegion.page: a count must be an int, got bool"):
            replace(REGION, page=True)
        with pytest.raises(ValueError, match="recovered: a count must be an int, got bool"):
            digitization(recovered=True)

    def test_an_int_is_refused_wherever_a_coordinate_is_meant(self) -> None:
        """Refused, not widened. A silent widening is a second idea of what a coordinate is."""
        with pytest.raises(ValueError, match="PlotRegion.x_start: a coordinate must be a float, got int"):
            replace(REGION, x_start=0)
        with pytest.raises(ValueError, match=r"MarkerOmission\('m07'\).y: a coordinate must be a float, got int"):
            replace(STRADDLER, y=3)

    def test_a_decimal_is_refused_wherever_a_coordinate_is_meant(self) -> None:
        with pytest.raises(ValueError, match="a coordinate must be a float, got Decimal"):
            replace(REGION, y_top=Decimal("640"))

    def test_a_bad_count_is_refused_before_it_can_balance_the_census(self) -> None:
        """``detected=True`` used to arithmetic as 1 and satisfy D9 outright."""
        with pytest.raises(ValueError, match="a count must be an int, got bool"):
            digitization(
                coverage=FigureCoverage.COMPLETE,
                census=MarkerCensus(detected=True),
                omissions=(),
                recovered=1,
            )

    def test_every_field_that_constructs_also_serializes(self) -> None:
        """The property the guards exist for, asserted directly.

        A record that constructs must serialize, and what it serializes must read back. Before
        the guards this chain broke in the middle for four different field values.
        """
        record = digitization()
        assert FigureDigitization.from_payload(digitization_record_payload(record)) == record

    def test_a_wrongly_typed_component_is_refused(self) -> None:
        for overrides, expected in (
            ({"plot_region": {"page": 4}}, "plot_region must be a PlotRegion"),
            ({"coverage": "partial"}, "coverage must be a FigureCoverage"),
            ({"omissions": [STRADDLER, OCCLUDED]}, "omissions must be a tuple"),
            ({"omissions": ("m07", OCCLUDED)}, r"omissions\[0\] must be a MarkerOmission"),
            ({"series_id": 3}, "series_id must be a string"),
        ):
            with pytest.raises(ValueError, match=expected):
                digitization(**overrides)

    def test_an_omission_with_a_foreign_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reason must be a MarkerOmissionReason"):
            MarkerOmission(marker_id="m01", reason="occluded", x=1.0, y=2.0)  # type: ignore[arg-type]

    def test_a_non_string_marker_id_is_refused_before_it_is_stripped(self) -> None:
        """``marker_id.strip()`` would raise ``AttributeError`` on an int, which is a crash
        rather than a refusal and tells a caller nothing about what it did wrong."""
        with pytest.raises(ValueError, match="marker_id must be a string, got int"):
            MarkerOmission(marker_id=7, reason=MarkerOmissionReason.OCCLUDED, x=1.0, y=2.0)  # type: ignore[arg-type]

    def test_a_non_string_detail_is_refused_on_both_types_that_carry_one(self) -> None:
        """``detail`` is free text, but it still has to SERIALIZE -- and a non-string would be
        refused by ``from_payload`` after reaching the store."""
        with pytest.raises(ValueError, match="MarkerOmission.*detail must be a string, got int"):
            MarkerOmission(marker_id="m01", reason=MarkerOmissionReason.OCCLUDED, x=1.0, y=2.0, detail=7)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="CensusUnavailable.detail must be a string, got int"):
            CensusUnavailable(reason=CensusUnavailableReason.DETECTOR_UNAVAILABLE, detail=7)  # type: ignore[arg-type]


class TestTheAuditabilityAxisHasExactlyTwoStates:
    """`UNCHECKABLE` used to return early before anything constrained `census`."""

    def test_a_foreign_census_object_is_refused(self) -> None:
        for foreign in (None, "unavailable", 12, {"kind": "counted", "detected": 12}, object()):
            with pytest.raises(ValueError, match="census must be a MarkerCensus or a CensusUnavailable"):
                digitization(coverage=FigureCoverage.UNCHECKABLE, census=foreign, omissions=())

    def test_the_refusal_is_not_reachable_only_through_the_auditable_branch(self) -> None:
        """The same refusal under a COMPLETE claim, so it is not the early return doing it."""
        with pytest.raises(ValueError, match="census must be a MarkerCensus or a CensusUnavailable"):
            digitization(coverage=FigureCoverage.COMPLETE, census=None, omissions=())

    def test_a_foreign_unavailability_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reason must be a CensusUnavailableReason"):
            CensusUnavailable(reason="detector_unavailable")  # type: ignore[arg-type]


class TestEveryLedgerEntryIsALoss:
    """Finding 4: coverage keys off ledger-emptiness, so a vacuous entry would lie.

    An earlier draft carried ``OUTSIDE_PLOT_REGION`` for a mark ruled out as a legend key, whose
    own description conceded that nothing was lost. A digitization that recovered every marker
    and merely noted one would have reported PARTIAL -- a record UNDERSTATING its completeness,
    as untrue as one overstating it. The member is gone and the ledger means one thing.
    """

    def test_no_reason_describes_a_marker_that_was_never_a_candidate(self) -> None:
        assert not hasattr(MarkerOmissionReason, "OUTSIDE_PLOT_REGION")
        assert "outside_plot_region" not in {reason.value for reason in MarkerOmissionReason}

    def test_every_surviving_reason_names_a_marker_this_series_should_have_had(self) -> None:
        assert {reason.value for reason in MarkerOmissionReason} == {
            "axis_boundary_straddle",
            "occluded",
            "series_ambiguous",
            "coordinate_unresolved",
        }

    def test_a_lossless_digitization_can_never_be_forced_to_report_partial(self) -> None:
        """For every reason there is, an entry carrying it means a marker really is missing --
        so `recovered` is strictly below the census total and PARTIAL is the true answer."""
        for reason in MarkerOmissionReason:
            record = digitization(
                census=MarkerCensus(detected=11),
                recovered=10,
                omissions=(replace(STRADDLER, reason=reason),),
            )
            assert record.coverage is FigureCoverage.PARTIAL
            assert record.recovered < record.census.detected

    def test_a_stored_payload_naming_the_removed_reason_is_refused(self) -> None:
        payload = digitization_record_payload(digitization())
        payload["omissions"][0]["reason"] = "outside_plot_region"
        with pytest.raises(ValueError, match="outside_plot_region"):
            FigureDigitization.from_payload(payload)


class TestANonCandidateMarkHasNowhereToBeSmuggled:
    """Removing ``OUTSIDE_PLOT_REGION`` took away a reason code, not the mark a producer meets.

    Without D6b the nearest available action -- call the legend key ``OCCLUDED``, record it at
    out-of-frame coordinates -- is silently accepted, and forces PARTIAL for a digitization that
    lost nothing. That is the very misclassification the removal was meant to prevent, so the
    containment rule has to be ENFORCED and not merely intended.
    """

    def test_an_omission_centred_outside_the_region_is_refused(self) -> None:
        far_away = replace(STRADDLER, x=9999.0, y=9999.0)
        with pytest.raises(ValueError, match="outside the plot region"):
            digitization(census=MarkerCensus(detected=11), recovered=10, omissions=(far_away,))

    def test_the_refusal_says_why_a_non_candidate_is_neither_counted_nor_ledgered(self) -> None:
        with pytest.raises(ValueError, match="neither counted in the census nor ledgered as a loss"):
            digitization(
                census=MarkerCensus(detected=11),
                recovered=10,
                omissions=(replace(STRADDLER, x=9999.0, y=9999.0),),
            )

    def test_no_reason_code_provides_a_way_around_it(self) -> None:
        """The workaround, tried with every label a producer could reach for."""
        for reason in MarkerOmissionReason:
            with pytest.raises(ValueError, match="outside the plot region"):
                digitization(
                    census=MarkerCensus(detected=11),
                    recovered=10,
                    omissions=(replace(STRADDLER, reason=reason, x=9999.0, y=9999.0),),
                )

    def test_each_edge_of_the_region_is_guarded(self) -> None:
        """A single-axis escape would leave three quarters of the plane reachable."""
        for x, y in ((71.9, 300.0), (520.1, 300.0), (300.0, 99.9), (300.0, 640.1)):
            with pytest.raises(ValueError, match="outside the plot region"):
                digitization(
                    census=MarkerCensus(detected=11),
                    recovered=10,
                    omissions=(replace(STRADDLER, x=x, y=y),),
                )

    def test_a_marker_centred_exactly_on_the_boundary_is_the_paradigm_straddle(self) -> None:
        """Containment is INCLUSIVE, and that is the whole point of the reason code: a marker
        centred on the axis is a candidate whose coordinate cannot be read, not an outsider."""
        for x, y in ((REGION.x_start, 300.0), (REGION.x_end, 300.0), (300.0, REGION.y_bottom), (300.0, REGION.y_top)):
            record = digitization(
                census=MarkerCensus(detected=11),
                recovered=10,
                omissions=(replace(STRADDLER, x=x, y=y),),
            )
            assert record.coverage is FigureCoverage.PARTIAL

    def test_the_module_docstring_states_the_rule_the_reason_code_points_at(self) -> None:
        """``MarkerOmissionReason`` refers a reader to a paragraph that has to exist."""
        module_doc = figure_digitization_record.__doc__
        assert module_doc is not None
        assert "WHAT COUNTS AS A CANDIDATE, AND WHERE A NON-CANDIDATE GOES" in module_doc
        for phrase in ("legend key", "DETECTION-log", "not counted in ``detected`` and is not ledgered"):
            assert phrase in module_doc, f"the docstring the reason code points at does not mention {phrase!r}"

    def test_the_pointer_from_the_reason_code_resolves(self) -> None:
        reason_doc = MarkerOmissionReason.__doc__
        assert reason_doc is not None
        assert "WHAT COUNTS AS A CANDIDATE, AND WHERE A NON-CANDIDATE GOES" in reason_doc


class TestTheAddressNamesTheClaimNotTheDigitization:
    """Finding 5, pinned as a tested fact rather than left as a warning nobody checks."""

    def test_records_agreeing_on_every_stored_field_share_one_address(self) -> None:
        """The limit, as far as it can honestly be demonstrated from inside the record.

        Renamed from a claim about "two DIFFERENT digitizations", which this cannot show and did
        not: the difference that matters lives in the recovered POINTS, and the record has no
        field for them, so two digitizations differing only there are indistinguishable HERE by
        construction. What is demonstrable is the mechanism -- the ledger's coordinates are
        hashed and the recovered points are represented by a bare count -- and
        ``test_the_payload_carries_a_count_and_not_one_recovered_coordinate`` below is what makes
        the collision structural rather than a fixture coincidence.
        """
        first = digitization(census=MarkerCensus(detected=11), recovered=10, omissions=(STRADDLER,))
        moved_omission = digitization(
            census=MarkerCensus(detected=11),
            recovered=10,
            omissions=(replace(STRADDLER, x=101.0, y=102.0),),
        )
        # The omission's own coordinates ARE hashed, so a ledger difference does move the address.
        assert compute_digitization_sha(digitization_record_payload(first)) != compute_digitization_sha(
            digitization_record_payload(moved_omission)
        )

        # What is NOT hashed is anything about the ten RECOVERED points. Two records agreeing on
        # every field the payload has share an address no matter what those ten points were.
        same_fields = digitization(census=MarkerCensus(detected=11), recovered=10, omissions=(STRADDLER,))
        assert compute_digitization_sha(digitization_record_payload(same_fields)) == compute_digitization_sha(
            digitization_record_payload(first)
        )

    def test_the_payload_carries_a_count_and_not_one_recovered_coordinate(self) -> None:
        """Why the collision above is structural rather than a fixture coincidence."""
        payload = digitization_record_payload(digitization())
        assert payload["recovered"] == 10
        assert "points" not in payload
        assert "series_sha256" not in payload
        assert "calibration" not in payload

    def test_the_limit_is_stated_where_a_caller_would_look(self) -> None:
        """Documented at the point of use, not only in the module docstring."""
        assert compute_digitization_sha.__doc__ is not None
        assert "ADDRESSES IS THE CLAIM, NOT THE DIGITIZATION" in compute_digitization_sha.__doc__


class TestCanonicalizationDoesNotDriftFromTheTableLane:
    """The contract the ticket names: mirror the table lane's, never invent a second one."""

    def test_coordinates_are_spelled_exactly_as_the_table_lane_spells_them(self) -> None:
        """Pinned against ``pdf_table_record``'s own private helper.

        Reaching into a private name is deliberate and is the point: the two copies exist
        because :mod:`carmel.schemas.datasets` imports FROM both modules and neither may import
        the other, and a duplicated definition is two definitions that agree until one is
        edited. This test is what makes that edit fail loudly.
        """
        from carmel.services.figure_digitization_record import _pt as figure_pt  # noqa: PLC0415
        from carmel.services.pdf_table_record import _pt as table_pt  # noqa: PLC0415

        for value in (0.0, -0.0, 1.0, 519.4, 0.1 + 0.2, 1e-300, 1e300, -72.5):
            assert figure_pt(value) == table_pt(value) == value.hex()

    def test_the_identifier_grammar_matches_the_schemas_own(self) -> None:
        """``series_id`` joins to ``Series.series_id``; two grammars would break the join."""
        from carmel.schemas.datasets import _IDENTIFIER_PATTERN  # noqa: PLC0415
        from carmel.services.figure_digitization_record import _IDENTIFIER_RE  # noqa: PLC0415

        assert _IDENTIFIER_RE.pattern == _IDENTIFIER_PATTERN.pattern
