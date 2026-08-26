"""Tests for the persisted cell-inventory record.

Every fixture is SYNTHETIC; no paper text enters the repository. The property under test is
that the record is a claim the DOCUMENT can refute -- that replay recomputes the grid rather
than reading it back, and that each way of failing to reach a verdict stays distinguishable
from the others.

The document these tests replay against is a real PDF built here in bytes, because
``verify_inventory_record`` takes bytes and runs the production fragment lane over them. A
stubbed extraction would test the comparison and skip the part that can actually drift.
"""

from __future__ import annotations

import hashlib
import inspect
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest

from carmel.services import pdf_tables
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.pdf_fragments import GlyphMapping, extract_fragments
from carmel.services.pdf_table_record import (
    _REFUSAL_DIAGNOSTIC_FIELDS,
    INVENTORY_PAYLOAD_KEYS,
    INVENTORY_PAYLOAD_VERSION,
    InventoryVerificationStatus,
    _comparable,
    compute_inventory_sha,
    footprint_unreadable_reason,
    inventory_code_sha256,
    inventory_record_payload,
    refusal_reasons_of,
    verify_inventory_record,
)
from carmel.services.pdf_tables import ClaimedFootprint, InventoryRefusalReason, build_inventory
from tests.pypdf_gate import require_pypdf


def _pdf(content: str) -> bytes:
    """A minimal one-page PDF whose content stream is ``content``.

    Hand-built rather than fixtured from a file so the geometry under test is visible in the
    test that depends on it, and so no paper is involved.
    """
    stream = zlib.compress(content.encode("ascii"))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    return bytes(out)


#: A caption and two three-column rows.
#:
#: The 14 pt line pitch is deliberate and load-bearing: the real target measures a 14.3 pt
#: caption-to-first-row gap over a 10.0 pt body pitch, and an earlier version of this fixture
#: spaced its rows 80 pt apart. That is not a table, and it made the bottom-edge guard look
#: broken when what was wrong was the geometry it was handed.
GRID = _pdf(
    "BT /F1 9 Tf 53 700 Td (Table 1 - conditions) Tj ET\n"
    "BT /F1 9 Tf 53 686 Td (Fuel) Tj ET\n"
    "BT /F1 9 Tf 123 686 Td (alpha) Tj ET\n"
    "BT /F1 9 Tf 223 686 Td (beta) Tj ET\n"
    "BT /F1 9 Tf 53 672 Td (phi) Tj ET\n"
    "BT /F1 9 Tf 123 672 Td (0.6) Tj ET\n"
    "BT /F1 9 Tf 223 672 Td (0.5) Tj ET\n"
)

FOOTPRINT = ClaimedFootprint(
    page=1,
    x_start=50.0,
    x_end=290.0,
    y_top=692.0,
    y_bottom=665.0,
    caption_text="Table 1 - conditions",
    caption_x_start=53.0,
    caption_baseline_y=700.0,
)


@contextmanager
def monkeypatched_unreadable_source() -> Iterator[None]:
    """Make :func:`inspect.getsource` fail the way a ``.pyc``-only install makes it fail.

    A real deployment shape, not a contrived one: source-stripped and zipped installs have no
    ``.py`` for ``inspect`` to read. The status must be its own, because degrading to a
    sentinel identity would let two records written under two DIFFERENT unknown versions
    compare equal.
    """
    original = inspect.getsource

    def refusing(obj: object) -> str:
        if obj is pdf_tables:
            raise OSError("could not get source code")
        return original(obj)  # type: ignore[arg-type]

    inspect.getsource = refusing  # type: ignore[assignment]
    try:
        yield
    finally:
        inspect.getsource = original  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _needs_pypdf() -> None:
    """Every test here runs a REAL extraction, so every one of them needs the engine.

    Autouse rather than module scope: the gate's own rule is to guard at the point of
    dependency, and an autouse fixture is that point for a module where the dependency is
    universal, while still leaving the skip attributable to each test rather than to an
    import. CI's base job installs without the ``agents`` extra, which is the environment
    this exists for.
    """
    require_pypdf()


@pytest.fixture(scope="module")
def record() -> dict:
    # The gate is repeated here, and the repetition is required rather than defensive: this
    # fixture is MODULE-scoped, and pytest sets a higher-scoped fixture up BEFORE the
    # function-scoped autouse gate above, so without this it raises before anything can skip.
    # CI's pypdf-free base job caught it; the local suite was green, which is the whole reason
    # that job exists.
    require_pypdf()
    extraction = extract_fragments(GRID)
    inventory = build_inventory(extraction, FOOTPRINT)
    assert inventory.refusals == (), f"fixture does not derive: {inventory.refusals}"
    return inventory_record_payload(inventory, raw_sha256=hashlib.sha256(GRID).hexdigest())


class TestTheRecordIsAClaimTheDocumentCanRefute:
    def test_a_record_reproduces_against_its_own_document(self, record: dict) -> None:
        result = verify_inventory_record(record, GRID)

        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.reproduced
        assert result.identity_moved == ()

    def test_replay_recomputes_the_grid_rather_than_reading_it_back(self, record: dict) -> None:
        """The whole point. Corrupt a derived row ordinal and replay must refuse it.

        If verification read the stored rows instead of recomputing them, this passes -- and
        the record would be a note it makes about itself rather than a claim the document
        can refute.
        """
        tampered = {**record, "rows": [{**record["rows"][0], "ordinal": 7}, *record["rows"][1:]]}

        result = verify_inventory_record(tampered, GRID)

        assert result.status is InventoryVerificationStatus.MISMATCHED
        assert "rows" in result.detail

    def test_a_shifted_footprint_does_not_reproduce_the_stored_grid(self, record: dict) -> None:
        """Replay re-derives from the STORED footprint, so moving it must change the result.

        This is the honest limit of the record: it proves the box still yields this grid, and
        never that the box is the right one.
        """
        moved = {**record, "footprint": {**record["footprint"], "y_bottom": (679.0).hex()}}

        result = verify_inventory_record(moved, GRID)

        assert result.status is InventoryVerificationStatus.MISMATCHED

    def test_a_refused_inventory_is_worth_storing_and_replays_as_refused(self) -> None:
        truncating = replace(FOOTPRINT, y_bottom=679.0)
        inventory = build_inventory(extract_fragments(GRID), truncating)
        assert inventory.refusals, "expected the truncated box to refuse"

        payload = inventory_record_payload(inventory, raw_sha256=hashlib.sha256(GRID).hexdigest())

        assert refusal_reasons_of(payload) == (InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX,)
        assert verify_inventory_record(payload, GRID).reproduced


class TestRefusalProseIsDiagnosticsAndTheReasonIsTheFinding:
    """``detail`` sat inside the compared bytes, so rewording one broke stored records.

    A record written before a diagnostic was rephrased came back MISMATCHED -- the status
    that means "this record is not evidence" -- with nothing about the document, the
    footprint or the finding changed. That is not a schema fault and does not move
    ``INVENTORY_PAYLOAD_VERSION``, which stamps the payload's SHAPE: the shape is identical
    and ``detail`` is still written. What was wrong is that prose was being asked to
    reproduce.
    """

    @staticmethod
    def _refused_payload() -> dict:
        truncating = replace(FOOTPRINT, y_bottom=679.0)
        inventory = build_inventory(extract_fragments(GRID), truncating)
        assert inventory.refusals, "expected the truncated box to refuse"
        return inventory_record_payload(inventory, raw_sha256=hashlib.sha256(GRID).hexdigest())

    def test_rewording_a_detail_does_not_break_a_stored_record(self) -> None:
        payload = self._refused_payload()
        reworded = {
            **payload,
            "refusals": [{**entry, "detail": "worded differently, same finding"} for entry in payload["refusals"]],
        }

        result = verify_inventory_record(reworded, GRID)

        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.identity_moved == (), "nothing about the derivation's identity moved either"

    def test_changing_a_REASON_still_mismatches(self) -> None:
        """The half that makes the exclusion a narrowing rather than a hole. ``reason`` names
        which check refused and is what a consumer switches on, so a record claiming a
        different one has not reproduced."""
        payload = self._refused_payload()
        relabelled = {
            **payload,
            "refusals": [{**entry, "reason": InventoryRefusalReason.EMPTY.value} for entry in payload["refusals"]],
        }

        result = verify_inventory_record(relabelled, GRID)

        assert result.status is InventoryVerificationStatus.MISMATCHED

    def test_detail_is_excluded_from_the_comparison_not_from_the_record(self) -> None:
        """A reader holding these bytes still gets the sentence, and the key set is
        unchanged -- which is the whole reason the version does not move."""
        payload = self._refused_payload()

        assert all(entry["detail"] for entry in payload["refusals"])
        assert set(payload) == INVENTORY_PAYLOAD_KEYS


class TestAMemberRefusalIsIdentifiedByItsOffendingGlyphs:
    """R-012 §3: ``reason`` + prose ``detail`` under-identified a member-specific refusal -- two
    refusals over DIFFERENT offending glyphs shared a reason and a count, so a stored refusal
    reproduced even when the glyph that caused it had changed. The refusal now carries
    ``member_digests``, structured identity that IS compared, the same way a successful cell's
    members are pinned."""

    @staticmethod
    def _impostor_payload(text: str) -> dict:
        from tests.test_pdf_tables import CAPTION, extraction_of, footprint, frag

        impostor = frag(text, 122.0, 146.0, 134.5, glyph_mapping=GlyphMapping.UNRESOLVED_IMPOSTOR)
        extraction = extraction_of(CAPTION, frag("Fuel", 53.0, 70.0, 134.5), impostor)
        inventory = build_inventory(extraction, footprint())
        assert [r.reason for r in inventory.refusals] == [InventoryRefusalReason.IMPOSTOR_MEMBER]
        return inventory_record_payload(inventory, raw_sha256="0" * 64)

    def test_the_refusal_records_its_offending_members_as_structured_identity(self) -> None:
        (refusal,) = self._impostor_payload("1.0")["refusals"]
        assert refusal["member_digests"], "an impostor refusal with no member identity is a prose count again"
        assert all(len(d) == 64 for d in refusal["member_digests"])

    def test_member_digests_is_compared_where_detail_is_not(self) -> None:
        assert "member_digests" not in _REFUSAL_DIAGNOSTIC_FIELDS
        assert "detail" in _REFUSAL_DIAGNOSTIC_FIELDS
        (compared,) = _comparable(self._impostor_payload("1.0"))["refusals"]
        assert "member_digests" in compared
        assert "detail" not in compared

    def test_two_refusals_over_different_glyphs_do_not_reproduce_each_other(self) -> None:
        one = self._impostor_payload("1.0")
        other = self._impostor_payload("9.9")

        # Same finding, same prose shape -- and yet distinguishable, because the offending
        # member digests differ and are part of what must reproduce.
        assert one["refusals"][0]["reason"] == other["refusals"][0]["reason"]
        assert one["refusals"][0]["member_digests"] != other["refusals"][0]["member_digests"]
        assert _comparable(one) != _comparable(other)


class TestEachWayOfNotReachingAVerdictStaysDistinct:
    def test_the_wrong_document_is_a_source_mismatch_not_an_inability_to_verify(self, record: dict) -> None:
        other = _pdf("BT /F1 9 Tf 53 700 Td (something else) Tj ET\n")

        result = verify_inventory_record(record, other)

        assert result.status is InventoryVerificationStatus.SOURCE_MISMATCH
        assert result.status is not InventoryVerificationStatus.MISMATCHED

    def test_an_unknown_payload_version_is_unreadable_not_mismatched(self, record: dict) -> None:
        result = verify_inventory_record({**record, "payload_version": 99}, GRID)

        assert result.status is InventoryVerificationStatus.PAYLOAD_UNREADABLE
        assert "99" in result.detail

    def test_a_malformed_payload_is_unreadable(self) -> None:
        result = verify_inventory_record({"payload_version": INVENTORY_PAYLOAD_VERSION}, GRID)

        assert result.status is InventoryVerificationStatus.PAYLOAD_UNREADABLE

    def test_an_unreadable_document_is_engine_unavailable_not_mismatched(self, record: dict) -> None:
        """Reached with bytes the fragment lane cannot read AT ALL.

        The record must claim those bytes as its own, or ``SOURCE_MISMATCH`` fires first and
        this branch is never entered -- which is how an unreachable status survives a suite
        that only asserts its members are distinct.
        """
        garbage = b"not a pdf at all"
        claiming_garbage = {**record, "raw_sha256": hashlib.sha256(garbage).hexdigest()}

        result = verify_inventory_record(claiming_garbage, garbage)

        assert result.status is InventoryVerificationStatus.ENGINE_UNAVAILABLE

    def test_every_status_is_reachable(self, record: dict) -> None:
        """Runs each scenario for real and collects what came back.

        Deliberately NOT a comparison of the enum against a hand-written set of its own
        members: that shape passes with every code path deleted, which this project has
        shipped once and caught once.
        """
        garbage = b"not a pdf at all"
        produced = {
            verify_inventory_record(payload, data).status
            for payload, data in (
                (record, GRID),
                ({**record, "rows": []}, GRID),
                (record, _pdf("BT /F1 9 Tf 53 700 Td (other) Tj ET\n")),
                ({**record, "payload_version": 99}, GRID),
                ({**record, "raw_sha256": hashlib.sha256(garbage).hexdigest()}, garbage),
            )
        }
        with monkeypatched_unreadable_source():
            produced.add(verify_inventory_record(record, GRID).status)

        assert produced == set(InventoryVerificationStatus), (
            f"unreachable: {set(InventoryVerificationStatus) - produced}"
        )


class TestIdentityIsReportedWithoutBlockingTheRecomputation:
    def test_a_moved_code_identity_still_recomputes_and_says_so(self, record: dict) -> None:
        """A stale identity must not short-circuit the check.

        Refusing to look once any identity moved is fail-closed but too coarse: several
        fragment-lane supersessions moved no coordinate at all. Reporting REPRODUCED with the
        drift named is a stronger result than declining to recompute.
        """
        stale = {**record, "inventory_code_sha256": "0" * 64}

        result = verify_inventory_record(stale, GRID)

        assert result.identity_moved == ("inventory_code",)
        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.reproduced

    def test_a_moved_fragment_geometry_still_recomputes_and_says_so(self, record: dict) -> None:
        """The geometry-engine identity is excluded from the bytes, exactly like the code one.

        ``fragment_geometry_sha256`` is one of ``_IDENTITY_FIELDS``, so a record written under a
        different geometry engine must still recompute its grid and report the drift, not come
        back an indistinguishable MISMATCHED. This is the test that fails if that field is
        dropped from the exclusion set and so creeps back into the compared bytes.
        """
        stale = {**record, "fragment_geometry_sha256": "0" * 64}

        result = verify_inventory_record(stale, GRID)

        assert result.identity_moved == ("fragment_geometry",)
        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.reproduced

    def test_a_moved_pypdf_version_still_recomputes_and_says_so(self, record: dict) -> None:
        """``pypdf_version`` is identity too, and its exclusion needs its own witness.

        A record written under a different ``pypdf`` build must reproduce with the version
        drift named, rather than being read as a grid that no longer reproduces. This is the
        test that fails if ``pypdf_version`` is dropped from the exclusion set and so creeps
        back into the compared bytes.
        """
        stale = {**record, "pypdf_version": "0.0.0-not-a-real-pypdf"}

        result = verify_inventory_record(stale, GRID)

        assert result.identity_moved == ("pypdf_version",)
        assert result.status is InventoryVerificationStatus.REPRODUCED
        assert result.reproduced

    def test_reproduced_under_drift_is_not_the_same_answer_as_reproduced(self, record: dict) -> None:
        """A caller that needs "same grid AND same code" must be able to tell them apart.

        This is the pairing that was UNREACHABLE before review: with the identity fields left
        inside the byte comparison, any drift forced MISMATCHED, so the module's own promise
        that REPRODUCED-with-drift is a stronger result than refusing to look could never be
        kept. The earlier version of this test asserted the contradiction instead of catching
        it.
        """
        clean = verify_inventory_record(record, GRID)
        drifted = verify_inventory_record({**record, "inventory_code_sha256": "0" * 64}, GRID)

        assert clean.reproduced and drifted.reproduced
        assert clean.identity_moved == ()
        assert drifted.identity_moved == ("inventory_code",)

    def test_a_moved_identity_does_not_hide_a_moved_grid(self, record: dict) -> None:
        """Excluding identity from the comparison must not exclude anything else."""
        both = {
            **record,
            "inventory_code_sha256": "0" * 64,
            "rows": [{**record["rows"][0], "ordinal": 7}, *record["rows"][1:]],
        }

        result = verify_inventory_record(both, GRID)

        assert result.status is InventoryVerificationStatus.MISMATCHED
        assert result.identity_moved == ("inventory_code",)

    def test_the_code_identity_tracks_the_derivation_and_its_measured_constants(self) -> None:
        """It is computed from source, so it cannot be forgotten the way a version int can."""
        assert len(inventory_code_sha256()) == 64
        assert inventory_code_sha256() == inventory_code_sha256()


class TestTheDeclaredShapeIsTheShapeActuallyWritten:
    """``INVENTORY_PAYLOAD_KEYS`` is what a reader holding no document uses to
    tell a replayable record from bytes that could never be verified at all --
    :class:`~carmel.schemas.datasets.EmbeddedTableInventory` refuses anything
    whose top-level keys are not exactly this set.

    That makes it a hand-maintained mirror of :func:`inventory_record_payload`,
    which is a thing that drifts. If it drifts wider, unreplayable records get
    embedded; if it drifts narrower, every honest record is refused. Neither
    shows up anywhere else, because both sides would still agree with
    themselves.
    """

    def test_a_real_record_has_exactly_the_declared_keys(self, record: dict) -> None:
        assert set(record) == set(INVENTORY_PAYLOAD_KEYS)

    def test_a_real_records_footprint_reads_back(self, record: dict) -> None:
        """The same pin, for the other half of "replayable".

        ``footprint_unreadable_reason`` is what the schema asks before letting a
        record be cited. If the builder ever renamed a footprint field or changed
        how a coordinate is serialized, every honest record would start being
        refused as unverifiable by construction -- and nothing else would catch
        it, because the verifier and the builder would still agree with each other.
        """
        assert footprint_unreadable_reason(record) is None

    def test_a_real_record_names_each_coordinate_once(self, record: dict) -> None:
        """The third pin, for the rule the schema now enforces on embedded records.

        ``EmbeddedTableInventory`` refuses a repeated ``(row, col)`` because a
        membership bit cannot express "present, but the record disagrees with
        itself about what is here". That rule is only safe if the builder never
        emits one -- otherwise the schema would refuse honest records, and neither
        side would notice, because each would still agree with itself.
        """
        coordinates = [(cell["row"], cell["col"]) for cell in record["cells"]]
        assert len(set(coordinates)) == len(coordinates)


class TestTheStoredFormIsCanonicalAndExact:
    def test_geometry_round_trips_bit_for_bit(self, record: dict) -> None:
        """``float.hex`` is exact; a decimal rendering would not have to be.

        Coordinates are the evidence, so a serialization that loses a bit loses the thing the
        record exists to pin.
        """
        for stored in record["column_bounds"]:
            for value in stored:
                assert float.fromhex(value).hex() == value

    def test_the_address_is_stable_and_key_order_does_not_change_it(self, record: dict) -> None:
        assert compute_inventory_sha(record) == compute_inventory_sha(dict(reversed(list(record.items()))))

    def test_the_payload_carries_no_float_so_the_canonical_store_accepts_it(self, record: dict) -> None:
        """``canonical_json_bytes`` rejects floats anywhere in the structure, by design."""
        assert canonical_json_bytes(record)

    def test_changing_any_coordinate_changes_the_address(self, record: dict) -> None:
        moved = {**record, "footprint": {**record["footprint"], "x_end": (291.0).hex()}}

        assert compute_inventory_sha(moved) != compute_inventory_sha(record)

    def test_a_malformed_raw_sha_is_refused_at_write_time(self) -> None:
        """It is the record's only link to the document and nothing re-derives it.

        A truncated or mistyped digest would mint a record that can never match any bytes,
        and every later check would report SOURCE_MISMATCH -- blaming the caller's file for
        the record's own defect.
        """
        inventory = build_inventory(extract_fragments(GRID), FOOTPRINT)

        for bad in ("", "abc", "A" * 64, "g" * 64, hashlib.sha256(GRID).hexdigest().upper()):
            with pytest.raises(ValueError, match="raw_sha256"):
                inventory_record_payload(inventory, raw_sha256=bad)

    def test_a_non_finite_coordinate_is_refused_rather_than_stored(self) -> None:
        """``float("nan").hex()`` is the valid string ``'nan'``.

        Without an explicit guard a nan coordinate round-trips into the store looking like a
        measurement, since the canonical layer only rejects float OBJECTS, not float-shaped
        strings this module produces.
        """
        inventory = build_inventory(extract_fragments(GRID), replace(FOOTPRINT, x_end=float("inf")))

        with pytest.raises(ValueError, match="non-finite"):
            inventory_record_payload(inventory, raw_sha256=hashlib.sha256(GRID).hexdigest())

    def test_cell_members_are_pinned_beyond_their_rendered_text(self, record: dict) -> None:
        """Text and an x-extent alone would not pin composition.

        Two different fragment sets can concatenate to the same string across the same span,
        so each member names its parent fragment's digest -- and, since one show can ground
        two cells, the glyph range and exact text piece the cell claims from it.
        """
        members = [m for c in record["cells"] for m in c["members"]]
        assert members, "expected at least one cell member"
        for member in members:
            assert set(member) == {"fragment_sha256", "glyph_start", "glyph_end", "text", "x_start", "x_end"}
            assert len(member["fragment_sha256"]) == 64
            assert member["text"]

    def test_a_version_1_record_is_unreadable_not_mismatched(self, record: dict) -> None:
        """The bump's contract: version 1 stored bare member digests, this code reads
        version 2, and "I cannot read this" must never degrade into "this does not
        reproduce". No v1 record exists in any workspace store (measured before the
        bump), so there is no versioned v1 verifier to keep honest -- just the refusal.
        """
        result = verify_inventory_record({**record, "payload_version": 1}, GRID)

        assert result.status is InventoryVerificationStatus.PAYLOAD_UNREADABLE
        assert "1" in result.detail

    def test_two_fragments_with_different_interiors_have_different_digests(self) -> None:
        """The demonstrated collision the v2 identity closes: same text, same outer
        extents, different interiors -- one drawn as a contiguous run, one with an
        internal spacing gap -- must not share a member identity."""
        from dataclasses import replace as _replace

        from carmel.services.pdf_table_record import _fragment_digest

        extraction = extract_fragments(GRID)
        fragment = next(f for f in extraction.fragments if f.text == "phi")
        assert fragment.glyph_intervals is not None
        pieces = list(fragment.glyph_intervals)
        # Shift one interior glyph's interval without touching text or outer extents.
        piece, start, end = pieces[1]
        pieces[1] = (piece, start + 0.5, end + 0.5)
        gapped = _replace(fragment, glyph_intervals=tuple(pieces))

        assert _fragment_digest(gapped) != _fragment_digest(fragment)

    def test_a_fragment_without_evidence_digests_differently_from_one_with_it(self) -> None:
        """`None` (unrecorded) is not an empty recording; the digest must keep them
        apart or a stripped fragment could impersonate a measured one."""
        from dataclasses import replace as _replace

        from carmel.services.pdf_table_record import _fragment_digest

        extraction = extract_fragments(GRID)
        fragment = next(f for f in extraction.fragments if f.text == "phi")

        assert _fragment_digest(_replace(fragment, glyph_intervals=None)) != _fragment_digest(fragment)
