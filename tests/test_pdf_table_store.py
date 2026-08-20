"""Tests for the on-disk inventory store.

Every fixture is SYNTHETIC; no paper text enters the repository. The document is the same
hand-built PDF ``tests.test_pdf_table_record`` uses, imported rather than copied so the two
modules cannot drift into testing two different geometries.

The property under test is the one the store exists for: a payload that came off DISK, put to
a document that came off disk, still has to re-derive its grid. Until this module existed the
verifier's only possible input was the same process's own output -- and a layer that
reproduces its own output has proved nothing. So the tests here are written to break that
loop, not to confirm it: the sharp ones write a WELL-ADDRESSED record whose claim is false and
require the document to catch it, and file a valid record under the wrong document and require
the layout to catch that.

``raw.bin`` is written through the production ``artifact_dir`` helper rather than at a path
spelled out here, so the layout under test comes from the code and not from this file's belief
about it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from carmel.services.evidence import artifact_dir
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import (
    InventoryVerificationStatus,
    compute_inventory_sha,
    inventory_record_bytes,
    inventory_record_payload,
)
from carmel.services.pdf_table_store import (
    InventoryStoreError,
    StoredInventoryOutcome,
    inventory_record_path,
    load_inventory_record,
    scan_inventory_records,
    store_inventory_record,
    verify_stored_inventory,
)
from carmel.services.pdf_tables import ClaimedFootprint, InventoryRefusalReason, build_inventory
from tests.pypdf_gate import require_pypdf
from tests.test_pdf_table_record import FOOTPRINT, GRID

#: Well above the fixture PDF and well under anything that would matter, so a test that means
#: to exercise the cap sets its own value rather than relying on this one.
MAX_BYTES = 10_000_000

RAW_SHA = hashlib.sha256(GRID).hexdigest()

#: A second, unrelated document. Its bytes only ever need to be DIFFERENT and hashable.
OTHER = b"%PDF-1.4\nnot the same document\n"
OTHER_SHA = hashlib.sha256(OTHER).hexdigest()


@pytest.fixture(autouse=True)
def _needs_pypdf() -> None:
    """Every test here derives an inventory from a real PDF, so every one needs the engine."""
    require_pypdf()


def _place_raw(workspace: Path, data: bytes) -> str:
    """Write ``data`` as ``raw.bin`` under its own content address, and return that address."""
    digest = hashlib.sha256(data).hexdigest()
    directory = artifact_dir(workspace, digest)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.bin").write_bytes(data)
    return digest


def _inventory(footprint: ClaimedFootprint = FOOTPRINT):  # type: ignore[no-untyped-def]
    return build_inventory(extract_fragments(GRID), footprint)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    _place_raw(tmp_path, GRID)
    return tmp_path


def _write_record_bytes(workspace: Path, raw_sha256: str, data: bytes) -> str:
    """Place ``data`` at the address its own bytes hash to, bypassing the store's writer.

    Used only by tests that must produce a file the store's own API cannot produce -- a
    non-canonical encoding, a moved record, a forged claim. Nothing here is a shortcut for the
    happy path, which always goes through :func:`store_inventory_record`.
    """
    address = hashlib.sha256(data).hexdigest()
    path = inventory_record_path(workspace, raw_sha256, address)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return address


class TestTheStoreClosesTheLoopTheVerifierCouldNot:
    def test_a_stored_record_verifies_against_the_document_off_disk(self, workspace: Path) -> None:
        """The end-to-end that did not exist: payload from disk, bytes from disk, grid re-derived."""
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.DERIVED
        assert result.verification is not None
        assert result.verification.status is InventoryVerificationStatus.REPRODUCED
        assert result.usable_as_table_evidence

    def test_the_stored_address_is_the_one_the_verifier_computes(self, workspace: Path) -> None:
        """One definition of the byte form, or the file drifts from the name it is filed under."""
        inventory = _inventory()
        address = store_inventory_record(workspace, inventory, raw_sha256=RAW_SHA)

        expected = compute_inventory_sha(inventory_record_payload(inventory, raw_sha256=RAW_SHA))

        assert address == expected

    def test_the_file_on_disk_is_exactly_the_bytes_the_address_is_over(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        on_disk = inventory_record_path(workspace, RAW_SHA, address).read_bytes()

        assert hashlib.sha256(on_disk).hexdigest() == address

    def test_re_storing_the_same_inventory_is_an_idempotent_no_op(self, workspace: Path) -> None:
        first = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        before = inventory_record_path(workspace, RAW_SHA, first).read_bytes()

        second = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

        assert second == first
        assert inventory_record_path(workspace, RAW_SHA, first).read_bytes() == before

    def test_a_loaded_payload_round_trips_the_one_that_was_stored(self, workspace: Path) -> None:
        inventory = _inventory()
        address = store_inventory_record(workspace, inventory, raw_sha256=RAW_SHA)

        loaded = load_inventory_record(workspace, RAW_SHA, address)

        assert loaded == inventory_record_payload(inventory, raw_sha256=RAW_SHA)

    def test_a_symlink_at_the_destination_is_never_reported_as_an_idempotent_store(self, workspace: Path) -> None:
        """Comparing bytes through a link would call it a success with no record published.

        ``os.link`` fails because a name is already taken; reading THROUGH that name
        follows the link, so a link aimed at any file holding these exact bytes would
        return the address as though it had been written -- while the address stays
        occupied by a link forever, under an append-only rule.
        """
        payload = inventory_record_payload(_inventory(), raw_sha256=RAW_SHA)
        address = compute_inventory_sha(payload)
        target = workspace / "decoy.json"
        target.write_bytes(inventory_record_bytes(payload))
        path = inventory_record_path(workspace, RAW_SHA, address)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)

        with pytest.raises(InventoryStoreError, match="is a symlink"):
            store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

    def test_no_temp_file_survives_a_successful_store(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        directory = inventory_record_path(workspace, RAW_SHA, address).parent

        assert [p.name for p in directory.iterdir()] == [f"{address}.json"]


class TestAWellAddressedRecordCanStillBeAFalseClaim:
    """The sharp half. An address proves the FILE; only the document proves the CLAIM."""

    def test_a_forged_grid_is_addressed_correctly_and_still_refuted(self, workspace: Path) -> None:
        """A record claiming a row ordinal the page does not support, stored at its OWN true address.

        Every integrity check the store makes passes: the bytes hash to their name, they are
        canonical, and the payload names the document it sits under. Nothing but re-deriving
        the grid from ``raw.bin`` can catch this, which is why the address is not the evidence.
        """
        payload = inventory_record_payload(_inventory(), raw_sha256=RAW_SHA)
        forged = {**payload, "rows": [{**payload["rows"][0], "ordinal": 7}, *payload["rows"][1:]]}
        address = _write_record_bytes(workspace, RAW_SHA, inventory_record_bytes(forged))

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.DERIVED
        assert result.verification is not None
        assert result.verification.status is InventoryVerificationStatus.MISMATCHED
        assert not result.usable_as_table_evidence

    def test_a_record_that_did_not_reproduce_reports_no_refusals_at_all(self, workspace: Path) -> None:
        """``reproduced_refusals`` must stay empty for an unproven record, and must not read as "none".

        The field is named for its precondition precisely because an empty tuple here means
        "not established", never "this record carries no refusals".
        """
        payload = inventory_record_payload(_inventory(), raw_sha256=RAW_SHA)
        forged = {**payload, "rows": [{**payload["rows"][0], "ordinal": 7}, *payload["rows"][1:]]}
        address = _write_record_bytes(workspace, RAW_SHA, inventory_record_bytes(forged))

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.reproduced_refusals == ()
        assert not result.usable_as_table_evidence

    def test_a_valid_record_filed_under_the_wrong_document_is_corrupt_not_absent(self, tmp_path: Path) -> None:
        """The guard nesting buys, and the reason the payload's ``raw_sha256`` is not redundant.

        The record's bytes are untouched and still hash to their own name, so every check that
        looks only at the file passes. Only the directory it sits in disagrees with what it
        says about itself -- and reporting that as ``None`` would let a corpus pass conclude
        this document has nothing stored for it.
        """
        _place_raw(tmp_path, GRID)
        _place_raw(tmp_path, OTHER)
        address = store_inventory_record(tmp_path, _inventory(), raw_sha256=RAW_SHA)
        moved = inventory_record_path(tmp_path, OTHER_SHA, address)
        moved.parent.mkdir(parents=True, exist_ok=True)
        moved.write_bytes(inventory_record_path(tmp_path, RAW_SHA, address).read_bytes())

        with pytest.raises(InventoryStoreError, match="moved or forged"):
            load_inventory_record(tmp_path, OTHER_SHA, address)

    def test_a_record_stored_for_one_document_is_invisible_under_another(self, tmp_path: Path) -> None:
        _place_raw(tmp_path, GRID)
        _place_raw(tmp_path, OTHER)
        store_inventory_record(tmp_path, _inventory(), raw_sha256=RAW_SHA)

        scan = scan_inventory_records(tmp_path, OTHER_SHA)

        assert scan.addresses == ()
        assert scan.every_entry_could_be_a_record


class TestAFileThatIsNotTheRecordItsAddressNames:
    def test_absence_is_none_and_never_an_error(self, workspace: Path) -> None:
        assert load_inventory_record(workspace, RAW_SHA, "0" * 64) is None

    def test_a_tampered_byte_refuses_rather_than_loading(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        path = inventory_record_path(workspace, RAW_SHA, address)
        path.write_bytes(path.read_bytes().replace(b'"page"', b'"pagE"'))

        with pytest.raises(InventoryStoreError, match="does not hash to its own address"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_non_canonical_encoding_that_hashes_correctly_is_refused(self, workspace: Path) -> None:
        """Hashing to your own name is cheap: any bytes do. Being the encoding this store wrote is not."""
        payload = inventory_record_payload(_inventory(), raw_sha256=RAW_SHA)
        pretty = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        address = _write_record_bytes(workspace, RAW_SHA, pretty)

        with pytest.raises(InventoryStoreError, match="not in canonical form"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_json_array_at_a_valid_address_is_refused(self, workspace: Path) -> None:
        address = _write_record_bytes(workspace, RAW_SHA, b"[1,2,3]")

        with pytest.raises(InventoryStoreError, match="not a JSON object"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_bytes_that_are_not_json_at_all_are_refused(self, workspace: Path) -> None:
        address = _write_record_bytes(workspace, RAW_SHA, b"not json")

        with pytest.raises(InventoryStoreError, match="not valid JSON"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_symlinked_record_pointing_out_of_the_workspace_is_refused(
        self, workspace: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """The address is caller-supplied, so a symlink at one reaches anywhere readable.

        The bytes it points at hash to the name it is filed under, so the digest check
        would have passed.
        """
        outside = tmp_path_factory.mktemp("elsewhere") / "outside.json"
        outside.write_bytes(b"{}")
        address = hashlib.sha256(b"{}").hexdigest()
        path = inventory_record_path(workspace, RAW_SHA, address)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(outside)

        with pytest.raises(InventoryStoreError, match="is a symlink"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_symlink_that_never_leaves_the_record_directory_is_still_refused(self, workspace: Path) -> None:
        """The one a containment test cannot see, and the reason a symlink is refused for BEING one.

        ``A.json -> B.json`` inside the same directory escapes nothing. B's bytes hash to
        B, so naming the link A makes the digest check irrelevant, and one file ends up
        answering for two addresses.
        """
        real = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        alias = hashlib.sha256(b"a different address entirely").hexdigest()
        link = inventory_record_path(workspace, RAW_SHA, alias)
        link.symlink_to(inventory_record_path(workspace, RAW_SHA, real))

        with pytest.raises(InventoryStoreError, match="is a symlink"):
            load_inventory_record(workspace, RAW_SHA, alias)

    def test_a_symlinked_record_is_a_corrupt_store_not_a_caller_error(self, workspace: Path) -> None:
        """It must reach a verdict, not raise through one: both digests were already valid."""
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        elsewhere = workspace / "moved.json"
        elsewhere.write_bytes(inventory_record_path(workspace, RAW_SHA, address).read_bytes())
        path = inventory_record_path(workspace, RAW_SHA, address)
        path.unlink()
        path.symlink_to(elsewhere)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.RECORD_CORRUPT
        assert "is a symlink" in result.detail

    def test_a_directory_at_a_valid_address_is_refused(self, workspace: Path) -> None:
        address = "1" * 64
        inventory_record_path(workspace, RAW_SHA, address).mkdir(parents=True)

        with pytest.raises(InventoryStoreError, match="does not resolve to a regular file"):
            load_inventory_record(workspace, RAW_SHA, address)


class TestOneDocumentsStoreNeverResolvesIntoAnothers:
    """A directory symlink aliases whole documents, and containment cannot see it either."""

    def test_a_records_directory_symlinked_to_another_document_is_refused(self, tmp_path: Path) -> None:
        _place_raw(tmp_path, GRID)
        _place_raw(tmp_path, OTHER)
        address = store_inventory_record(tmp_path, _inventory(), raw_sha256=RAW_SHA)
        real_dir = inventory_record_path(tmp_path, RAW_SHA, address).parent
        aliased = inventory_record_path(tmp_path, OTHER_SHA, address).parent
        aliased.parent.mkdir(parents=True, exist_ok=True)
        aliased.symlink_to(real_dir, target_is_directory=True)

        with pytest.raises(InventoryStoreError, match="not its own directory"):
            load_inventory_record(tmp_path, OTHER_SHA, address)

    def test_an_aliased_records_directory_is_refused_on_the_write_path_too(self, tmp_path: Path) -> None:
        """Otherwise a store for B silently publishes into A's directory."""
        _place_raw(tmp_path, GRID)
        _place_raw(tmp_path, OTHER)
        address = store_inventory_record(tmp_path, _inventory(), raw_sha256=RAW_SHA)
        aliased = inventory_record_path(tmp_path, OTHER_SHA, address).parent
        aliased.parent.mkdir(parents=True, exist_ok=True)
        aliased.symlink_to(inventory_record_path(tmp_path, RAW_SHA, address).parent, target_is_directory=True)

        with pytest.raises(InventoryStoreError, match="not its own directory"):
            store_inventory_record(tmp_path, _inventory(), raw_sha256=OTHER_SHA)

    def test_an_aliased_artifact_directory_never_reaches_a_verdict(self, tmp_path: Path) -> None:
        """Serving B's bytes as A's would produce a verdict about the wrong document.

        Which guard catches it is worth pinning rather than leaving loose. The whole
        artifact directory is the link, so ``table_inventories/`` under it resolves away
        too and the RECORD side refuses first -- ``verify_stored_inventory`` never gets as
        far as reading ``raw.bin``. The identical check inside ``_read_source_bytes`` is
        therefore unreachable from here and is kept anyway, so that function does not
        depend on its caller having checked first; asserting the loose "not DERIVED" here
        would have passed even with both guards deleted and the record simply absent.
        """
        _place_raw(tmp_path, GRID)
        address = store_inventory_record(tmp_path, _inventory(), raw_sha256=RAW_SHA)
        aliased = artifact_dir(tmp_path, OTHER_SHA)
        aliased.parent.mkdir(parents=True, exist_ok=True)
        aliased.symlink_to(artifact_dir(tmp_path, RAW_SHA), target_is_directory=True)

        result = verify_stored_inventory(tmp_path, OTHER_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.RECORD_CORRUPT
        assert "not its own directory" in result.detail
        assert not result.usable_as_table_evidence

    def test_a_symlinked_raw_bin_is_source_unavailable(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        raw_path = artifact_dir(workspace, RAW_SHA) / "raw.bin"
        copy = artifact_dir(workspace, RAW_SHA) / "raw-copy.bin"
        copy.write_bytes(raw_path.read_bytes())
        raw_path.unlink()
        raw_path.symlink_to(copy)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.SOURCE_UNAVAILABLE
        assert "is a symlink" in result.detail


class TestTheSizeCapHoldsOnBothSides:
    def test_the_store_refuses_to_mint_a_record_it_could_never_read_back(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read-only cap would publish unreachable garbage at a permanent address."""
        monkeypatch.setattr("carmel.services.pdf_table_store._MAX_RECORD_BYTES", 16)

        with pytest.raises(InventoryStoreError, match="could never be read back"):
            store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

        assert scan_inventory_records(workspace, RAW_SHA).addresses == ()

    def test_a_record_that_grows_after_its_size_is_taken_is_still_bounded(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The read is capped by what it reads, not by a stat that preceded it.

        ``stat``-then-read states a memory bound it does not hold: the file can grow in
        between. This asserts the bound against a file that is over the cap at read time
        regardless of what any earlier call would have reported.
        """
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        monkeypatch.setattr("carmel.services.pdf_table_store._MAX_RECORD_BYTES", 16)

        with pytest.raises(InventoryStoreError, match="larger than the 16-byte cap"):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_malformed_address_raises_rather_than_reading_a_path(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="invalid inventory_sha256"):
            load_inventory_record(workspace, RAW_SHA, "../../etc/passwd")

    def test_a_malformed_document_digest_raises(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="invalid raw_sha256"):
            load_inventory_record(workspace, "not-a-sha", "0" * 64)

    def test_an_address_with_a_trailing_newline_is_refused(self, workspace: Path) -> None:
        """``re.match`` would accept this; ``fullmatch`` is the reason it does not."""
        with pytest.raises(ValueError, match="invalid inventory_sha256"):
            load_inventory_record(workspace, RAW_SHA, "0" * 64 + "\n")


class TestTheRecordAndTheDocumentFailSeparately:
    def test_a_missing_document_is_source_unavailable_not_a_mismatch(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        (artifact_dir(workspace, RAW_SHA) / "raw.bin").unlink()

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.SOURCE_UNAVAILABLE
        assert result.verification is None
        assert not result.usable_as_table_evidence

    def test_a_document_that_does_not_hash_to_its_own_directory_is_source_corrupt(self, workspace: Path) -> None:
        """Distinct from unavailable: the bytes are here and are provably not this document."""
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        (artifact_dir(workspace, RAW_SHA) / "raw.bin").write_bytes(GRID + b"\n")

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.SOURCE_CORRUPT
        assert result.verification is None

    def test_an_oversized_document_is_source_unavailable(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=16)

        assert result.outcome is StoredInventoryOutcome.SOURCE_UNAVAILABLE
        assert "larger than the 16-byte cap" in result.detail

    def test_an_absent_record_is_reported_as_absent_never_as_a_refusal(self, workspace: Path) -> None:
        result = verify_stored_inventory(workspace, RAW_SHA, "0" * 64, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.RECORD_ABSENT
        assert result.verification is None
        assert not result.usable_as_table_evidence

    def test_a_corrupt_record_never_reaches_the_document(self, workspace: Path) -> None:
        address = _write_record_bytes(workspace, RAW_SHA, b"not json")

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.RECORD_CORRUPT
        assert result.verification is None

    def test_every_non_derived_outcome_withholds_the_evidence_verdict(self, workspace: Path) -> None:
        """One assertion per way of not reaching the document, so none of them can drift into True."""
        (artifact_dir(workspace, RAW_SHA) / "raw.bin").unlink()
        stored = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        outcomes = {
            verify_stored_inventory(workspace, RAW_SHA, stored, max_bytes=MAX_BYTES),
            verify_stored_inventory(workspace, RAW_SHA, "0" * 64, max_bytes=MAX_BYTES),
            verify_stored_inventory(
                workspace, RAW_SHA, _write_record_bytes(workspace, RAW_SHA, b"not json"), max_bytes=MAX_BYTES
            ),
        }

        assert {r.outcome for r in outcomes} == {
            StoredInventoryOutcome.SOURCE_UNAVAILABLE,
            StoredInventoryOutcome.RECORD_ABSENT,
            StoredInventoryOutcome.RECORD_CORRUPT,
        }
        assert not any(r.usable_as_table_evidence for r in outcomes)


class TestARefusalIsStorableAndNeverReadsAsEvidence:
    """Today no table in the corpus yields a complete inventory, so a store that would only
    hold complete ones would be a store with nothing in it. The whole burden falls on the one
    predicate that separates "the document reproducibly refuses this box" from "this box is a
    table"."""

    def test_a_refused_inventory_stores_and_reproduces_like_any_other(self, workspace: Path) -> None:
        cut = replace(FOOTPRINT, y_bottom=680.0)
        inventory = _inventory(cut)
        assert inventory.refusals, "fixture must refuse for this test to mean anything"

        address = store_inventory_record(workspace, inventory, raw_sha256=RAW_SHA)
        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.outcome is StoredInventoryOutcome.DERIVED
        assert result.verification is not None
        assert result.verification.status is InventoryVerificationStatus.REPRODUCED

    def test_a_reproduced_refusal_is_not_table_evidence(self, workspace: Path) -> None:
        """``REPRODUCED`` alone would say yes here. Reproducing a refusal proves the refusal."""
        address = store_inventory_record(workspace, _inventory(replace(FOOTPRINT, y_bottom=680.0)), raw_sha256=RAW_SHA)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.reproduced_refusals == (InventoryRefusalReason.ORPHANED_BAND_BELOW_THE_BOX,)
        assert not result.usable_as_table_evidence

    def test_a_second_refusal_reason_reaches_the_reader_intact(self, workspace: Path) -> None:
        """A different refusal must not collapse into the first one on the way out of the store."""
        address = store_inventory_record(workspace, _inventory(replace(FOOTPRINT, x_end=200.0)), raw_sha256=RAW_SHA)

        result = verify_stored_inventory(workspace, RAW_SHA, address, max_bytes=MAX_BYTES)

        assert result.reproduced_refusals == (InventoryRefusalReason.TRUNCATED_COLUMN_BESIDE_THE_BOX,)
        assert not result.usable_as_table_evidence

    def test_the_two_halves_of_the_evidence_test_fail_independently(self, workspace: Path) -> None:
        """Neither half alone decides it, so neither can be dropped without the other noticing.

        Reproduced-with-refusals and unrefused-but-not-reproduced both exist here, and both
        must be False. A predicate testing only one of them passes one of these and fails the
        other, which is what makes this pair a test rather than a restatement.
        """
        refused = store_inventory_record(workspace, _inventory(replace(FOOTPRINT, y_bottom=680.0)), raw_sha256=RAW_SHA)
        payload = inventory_record_payload(_inventory(), raw_sha256=RAW_SHA)
        forged = {**payload, "rows": [{**payload["rows"][0], "ordinal": 7}, *payload["rows"][1:]]}
        unproven = _write_record_bytes(workspace, RAW_SHA, inventory_record_bytes(forged))

        reproduced_but_refused = verify_stored_inventory(workspace, RAW_SHA, refused, max_bytes=MAX_BYTES)
        unrefused_but_unproven = verify_stored_inventory(workspace, RAW_SHA, unproven, max_bytes=MAX_BYTES)

        assert reproduced_but_refused.verification is not None
        assert reproduced_but_refused.verification.status is InventoryVerificationStatus.REPRODUCED
        assert reproduced_but_refused.reproduced_refusals != ()
        assert not reproduced_but_refused.usable_as_table_evidence

        assert unrefused_but_unproven.verification is not None
        assert unrefused_but_unproven.verification.status is not InventoryVerificationStatus.REPRODUCED
        assert unrefused_but_unproven.reproduced_refusals == ()
        assert not unrefused_but_unproven.usable_as_table_evidence


class TestScanReportsWhatItCouldNotRead:
    def test_a_document_with_no_records_scans_empty_and_complete(self, workspace: Path) -> None:
        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == ()
        assert scan.every_entry_could_be_a_record

    def test_a_stored_record_is_listed(self, workspace: Path) -> None:
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == (address,)
        assert scan.every_entry_could_be_a_record

    def test_an_unexplained_file_is_a_problem_rather_than_debris(self, workspace: Path) -> None:
        store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        (inventory_record_path(workspace, RAW_SHA, "0" * 64).parent / "notes.txt").write_bytes(b"hi")

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert not scan.every_entry_could_be_a_record
        assert [p.detail for p in scan.problems] == ["notes.txt is not a <64-hex>.json inventory record"]

    def test_a_sha_shaped_directory_is_a_problem_not_an_address(self, workspace: Path) -> None:
        """Name-shaped is not record-shaped; listing it would hand a producer a dead address."""
        address = "1" * 64
        inventory_record_path(workspace, RAW_SHA, address).mkdir(parents=True)

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == ()
        assert not scan.every_entry_could_be_a_record
        assert scan.problems[0].address == address

    def test_a_sha_shaped_symlink_is_a_problem_not_an_address(self, workspace: Path) -> None:
        real = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        alias = "2" * 64
        inventory_record_path(workspace, RAW_SHA, alias).symlink_to(inventory_record_path(workspace, RAW_SHA, real))

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == (real,)
        assert [p.address for p in scan.problems] == [alias]

    def test_a_leftover_temp_file_is_skipped_and_cannot_hide_a_record(self, workspace: Path) -> None:
        """A ``.``-prefixed name can never be a 64-hex address, so skipping one hides nothing."""
        address = store_inventory_record(workspace, _inventory(), raw_sha256=RAW_SHA)
        (inventory_record_path(workspace, RAW_SHA, address).parent / f".{address}.abc.tmp").write_bytes(b"")

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == (address,)
        assert scan.every_entry_could_be_a_record

    def test_a_scan_does_not_open_or_verify_anything(self, workspace: Path) -> None:
        """Listed is not loaded: a corrupt record still shows up, and still has to survive loading."""
        address = _write_record_bytes(workspace, RAW_SHA, b"not json")

        scan = scan_inventory_records(workspace, RAW_SHA)

        assert scan.addresses == (address,)
        assert scan.every_entry_could_be_a_record
        with pytest.raises(InventoryStoreError):
            load_inventory_record(workspace, RAW_SHA, address)

    def test_a_malformed_document_digest_raises_rather_than_becoming_a_problem(self, workspace: Path) -> None:
        """A typo is a fact about the caller; turning it into a per-document problem would read
        as a merely unreadable paper forever."""
        with pytest.raises(ValueError, match="invalid raw_sha256"):
            scan_inventory_records(workspace, "nope")
