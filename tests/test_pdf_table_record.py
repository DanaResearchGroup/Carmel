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
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import (
    INVENTORY_PAYLOAD_KEYS,
    INVENTORY_PAYLOAD_VERSION,
    InventoryVerificationStatus,
    compute_inventory_sha,
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
        so the member digests are what make the cell's makeup checkable.
        """
        cells = [c for c in record["cells"] if c["member_digests"]]
        assert cells, "expected at least one cell with members"
        assert all(len(d) == 64 for c in cells for d in c["member_digests"])
