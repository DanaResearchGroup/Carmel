"""Tests for carmel.services.extraction_record: content-addressed extraction records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from carmel.services import semantic_deps
from carmel.services.extraction_record import (
    ExtractionRecordError,
    UnknownPypdfVersionError,
    compute_extraction_sha,
    current_extraction_records,
    list_extraction_records,
    load_extraction_record,
    store_extraction_record,
    stored_extraction_sha256,
    verify_extraction_record,
)

RAW_SHA = "a" * 64
OTHER_RAW_SHA = "b" * 64


def _extracted_text_bytes(text: str = "hello", *, extractor: str = "pdf:pypdf") -> bytes:
    """Build synthetic, VALID ``ExtractedText`` JSON bytes (extra="forbid" satisfied)."""
    payload = {
        "text": text,
        "normalized": text.lower(),
        "sections": [{"label": "body", "start": 0, "end": len(text), "page": 1}],
        "page_count": 1,
        "extractor": extractor,
        "lossy": False,
        "page_failures": [],
    }
    return json.dumps(payload).encode("utf-8")


def _identity_payload(**overrides: str) -> dict[str, str]:
    payload = {
        "identity_payload_version": "2",
        "parent_raw_sha256": RAW_SHA,
        "extractor": "pdf:pypdf",
        "extractor_code_sha256": "c" * 64,
        "pypdf_version": "5.1.0",
        "extracted_sha256": "d" * 64,
        "extracted_text_sha256": "e" * 64,
    }
    payload.update(overrides)
    return payload


def _store(
    tmp_path: Path,
    *,
    raw_sha256: str = RAW_SHA,
    extractor: str = "pdf:pypdf",
    extractor_code_sha256: str = "c" * 64,
    pypdf_version: str = "5.1.0",
    extracted_json_bytes: bytes | None = None,
    text: str = "hello",
) -> str:
    if extracted_json_bytes is None:
        extracted_json_bytes = _extracted_text_bytes(text, extractor=extractor)
    return store_extraction_record(
        tmp_path,
        raw_sha256=raw_sha256,
        extractor=extractor,
        extractor_code_sha256=extractor_code_sha256,
        pypdf_version=pypdf_version,
        extracted_json_bytes=extracted_json_bytes,
    )


class TestComputeExtractionSha:
    """Direct unit tests of the hash helper alone.

    These are deliberately INSUFFICIENT on their own to prove the identity is
    actually bound into the stored address -- see
    :class:`TestStoredAddressIsBoundToTheExtractorIdentity` below, which drives the
    real store path instead.
    """

    def test_stable_and_independent_of_field_order(self) -> None:
        payload = _identity_payload()
        reordered = dict(reversed(list(payload.items())))

        assert compute_extraction_sha(payload) == compute_extraction_sha(reordered)

    def test_changing_extractor_code_sha256_changes_address(self) -> None:
        base = compute_extraction_sha(_identity_payload())
        changed = compute_extraction_sha(_identity_payload(extractor_code_sha256="f" * 64))

        assert base != changed

    def test_changing_pypdf_version_changes_address(self) -> None:
        base = compute_extraction_sha(_identity_payload())
        changed = compute_extraction_sha(_identity_payload(pypdf_version="5.2.0"))

        assert base != changed

    def test_rejects_payload_missing_a_required_field(self) -> None:
        payload = _identity_payload()
        del payload["extractor_code_sha256"]

        with pytest.raises(ValueError, match="malformed extraction identity payload"):
            compute_extraction_sha(payload)

    def test_rejects_payload_with_an_unexpected_extra_field(self) -> None:
        payload = _identity_payload(bogus_field="x")

        with pytest.raises(ValueError, match="malformed extraction identity payload"):
            compute_extraction_sha(payload)

    def test_rejects_payload_omitting_pypdf_version_for_a_pypdf_dependent_extractor(self) -> None:
        payload = _identity_payload()
        del payload["pypdf_version"]

        with pytest.raises(ValueError, match="malformed extraction identity payload"):
            compute_extraction_sha(payload)

    def test_rejects_payload_carrying_pypdf_version_for_a_non_pypdf_extractor(self) -> None:
        """A non-pypdf extractor's payload must OMIT pypdf_version entirely (F4)."""
        payload = _identity_payload(extractor="html")

        with pytest.raises(ValueError, match="malformed extraction identity payload"):
            compute_extraction_sha(payload)

    def test_rejects_non_sha_shaped_field_value(self) -> None:
        payload = _identity_payload(parent_raw_sha256="not-a-sha")

        with pytest.raises(ValueError, match="invalid parent_raw_sha256"):
            compute_extraction_sha(payload)

    def test_rejects_empty_string_field_value(self) -> None:
        payload = _identity_payload(identity_payload_version="")

        with pytest.raises(ValueError, match="malformed extraction identity payload field"):
            compute_extraction_sha(payload)


class TestStoredAddressIsBoundToTheExtractorIdentity:
    """The address a record is ACTUALLY STORED AT must depend on the extractor identity.

    The tests in :class:`TestComputeExtractionSha` above deliberately do NOT cover
    this. They hand :func:`compute_extraction_sha` two payload dicts that already
    differ and assert the digests differ -- which is a property of sha256, not of
    this module, and stays true even if the production path never puts the
    extractor identity into the payload at all. Hard-coding
    ``"extractor_code_sha256"`` to a constant inside ``_build_identity_payload``
    leaves all of those tests green while destroying the entire point of the
    milestone: an extraction produced by materially different Carmel code would
    address, and therefore silently collide with and be indistinguishable from, one
    produced by the old code.

    These tests drive the real :func:`store_extraction_record` path instead, so they
    fail in the direction of that defect.
    """

    def test_two_code_shas_store_to_two_distinct_addresses(self, tmp_path: Path) -> None:
        first = _store(tmp_path, extractor_code_sha256="a" * 64)
        second = _store(tmp_path, extractor_code_sha256="b" * 64)

        assert first != second
        # Both survive: an extraction record is append-only, never superseded in place.
        assert {record.extraction_sha256 for record in list_extraction_records(tmp_path, RAW_SHA)} == {first, second}

    def test_two_pypdf_versions_store_to_two_distinct_addresses(self, tmp_path: Path) -> None:
        first = _store(tmp_path, pypdf_version="5.1.0")
        second = _store(tmp_path, pypdf_version="5.2.0")

        assert first != second
        assert len(list_extraction_records(tmp_path, RAW_SHA)) == 2

    def test_identical_bytes_from_a_different_extractor_do_not_share_an_address(self, tmp_path: Path) -> None:
        """Same extracted text, different extractor: two records, not one.

        Without this, two extractors that happen to agree byte-for-byte on one
        document would be recorded as a single indistinguishable extraction.
        """
        first = _store(tmp_path, extractor="pdf:pypdf", text="same text")
        second = _store(tmp_path, extractor="html", text="same text")

        assert first != second

    def test_non_pypdf_extractor_address_is_stable_across_pypdf_versions(self, tmp_path: Path) -> None:
        """F4: a non-pypdf-dependent extractor's address must NOT move when pypdf_version differs."""
        first = _store(tmp_path, extractor="html", pypdf_version="5.1.0", text="same text")
        second = _store(tmp_path, extractor="html", pypdf_version="9.9.9", text="same text")

        assert first == second
        assert len(list_extraction_records(tmp_path, RAW_SHA)) == 1

    def test_pdf_unavailable_extractor_is_storable_without_a_known_pypdf_version(self, tmp_path: Path) -> None:
        """F4: pdf:unavailable must NOT be treated as pypdf-dependent -- it means pypdf is NOT importable."""
        extraction_sha = _store(
            tmp_path,
            extractor="pdf:unavailable",
            pypdf_version=semantic_deps._PYPDF_VERSION_UNKNOWN,
            text="no pdf backend",
        )

        record = load_extraction_record(tmp_path, RAW_SHA, extraction_sha)
        assert record is not None
        assert record.extractor == "pdf:unavailable"


class TestUnknownPypdfVersionRefused:
    def test_pdf_extractor_with_unknown_pypdf_version_raises_named_exception(self) -> None:
        payload = _identity_payload(pypdf_version=semantic_deps._PYPDF_VERSION_UNKNOWN)

        with pytest.raises(UnknownPypdfVersionError):
            compute_extraction_sha(payload)

    def test_non_pdf_extractor_with_unknown_pypdf_version_is_unaffected(self) -> None:
        payload = _identity_payload(extractor="html")
        del payload["pypdf_version"]

        # Must not raise: pypdf version is irrelevant provenance for a non-PDF extractor.
        compute_extraction_sha(payload)

    def test_store_extraction_record_refuses_unknown_pypdf_version_for_pdf_extractor(self, tmp_path: Path) -> None:
        with pytest.raises(UnknownPypdfVersionError):
            _store(tmp_path, pypdf_version=semantic_deps._PYPDF_VERSION_UNKNOWN)


class TestExtractedTextParsingAndCrossCheck:
    """F1: store_extraction_record parses ExtractedText from the bytes and cross-checks the extractor."""

    def test_rejects_bytes_that_are_not_valid_json(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionRecordError, match="not valid JSON"):
            _store(tmp_path, extracted_json_bytes=b"not json at all")

    def test_rejects_json_that_does_not_parse_as_extracted_text(self, tmp_path: Path) -> None:
        # Missing required fields (normalized, sections, extractor) and/or extra="forbid" violation.
        with pytest.raises(ExtractionRecordError, match="does not parse as ExtractedText"):
            _store(tmp_path, extracted_json_bytes=b'{"text": "hello"}')

    def test_rejects_extractor_argument_that_disagrees_with_the_parsed_bytes(self, tmp_path: Path) -> None:
        bytes_claiming_html = _extracted_text_bytes("hello", extractor="html")

        with pytest.raises(ExtractionRecordError, match="extractor mismatch"):
            _store(tmp_path, extractor="pdf:pypdf", extracted_json_bytes=bytes_claiming_html)

    def test_persisted_text_txt_is_the_parsed_text_field(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path, text="the parsed body text")

        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        assert dest.joinpath("text.txt").read_text(encoding="utf-8") == "the parsed body text"


class TestStoreExtractionRecordIdempotence:
    def test_exact_re_store_at_same_address_is_a_no_op(self, tmp_path: Path) -> None:
        first_sha = _store(tmp_path)
        second_sha = _store(tmp_path)

        assert first_sha == second_sha
        records = list_extraction_records(tmp_path, RAW_SHA)
        assert len(records) == 1

    def test_re_store_after_out_of_band_on_disk_corruption_of_the_same_address_raises(self, tmp_path: Path) -> None:
        """Renamed from the old, mislabeled ``..._differing_bytes_at_same_address_raises``.

        That name claimed to test "re-store with genuinely differing bytes at the
        same computed address" -- but that scenario is structurally near-unreachable
        honestly: extracted_sha256/extracted_text_sha256 are folded into the
        identity payload, so bytes that actually differ (almost always) hash to a
        genuinely DIFFERENT address, not the same one. What the test body actually
        does -- and all it can honestly do without an actual sha256 collision -- is
        tamper the ALREADY-STORED extracted.json out-of-band, then re-store the
        ORIGINAL bytes at the same computed address, and confirm the on-disk
        mismatch is caught rather than silently overwritten.
        """
        extraction_sha = _store(tmp_path, text="hello")

        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        (dest / "extracted.json").write_bytes(b'{"corrupted": true}')

        with pytest.raises(ExtractionRecordError, match="collision"):
            _store(tmp_path, text="hello")

    def test_genuinely_differing_extracted_json_at_the_same_computed_address_is_unreachable_honestly(
        self, tmp_path: Path
    ) -> None:
        """Documents the practical limit named in the class docstring above.

        Two calls that produce ACTUALLY different extracted.json bytes always
        differ in extracted_sha256 (folded into the identity payload), so they
        never land at the same address to begin with -- there is nothing further
        to assert here beyond confirming the two addresses genuinely differ.
        """
        first = _store(tmp_path, text="hello")
        second = _store(tmp_path, text="a genuinely different body")

        assert first != second


class TestListExtractionRecords:
    def test_lists_every_stored_record_for_a_raw_sha(self, tmp_path: Path) -> None:
        sha_one = _store(tmp_path, text="one")
        sha_two = _store(tmp_path, extractor_code_sha256="f" * 64, text="two")
        _store(tmp_path, raw_sha256=OTHER_RAW_SHA, text="unrelated")

        records = list_extraction_records(tmp_path, RAW_SHA)

        assert {r.extraction_sha256 for r in records} == {sha_one, sha_two}

    def test_no_records_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_extraction_records(tmp_path, RAW_SHA) == []

    def test_symlink_escaping_the_extractions_directory_is_skipped(self, tmp_path: Path) -> None:
        """F6: a sha-shaped symlink entry resolving outside extractions/ must never be followed."""
        _store(tmp_path, text="genuine")

        records_dir = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions"
        outside_target = tmp_path / "outside_target"
        outside_target.mkdir()
        (outside_target / "meta.json").write_text("{}", encoding="utf-8")
        escaping_name = "f" * 64
        (records_dir / escaping_name).symlink_to(outside_target, target_is_directory=True)

        records = list_extraction_records(tmp_path, RAW_SHA)

        assert escaping_name not in {r.extraction_sha256 for r in records}
        assert len(records) == 1


class TestStoredExtractionSha256:
    def test_matches_a_record_stored_with_todays_identity(self, tmp_path: Path) -> None:
        identity = semantic_deps.extraction_identity()
        extracted_json_bytes = _extracted_text_bytes("current")
        stored_sha = _store(
            tmp_path,
            extractor_code_sha256=identity.code_sha256,
            pypdf_version=identity.pypdf_version,
            extracted_json_bytes=extracted_json_bytes,
        )

        resolved = stored_extraction_sha256(
            tmp_path,
            RAW_SHA,
            extractor="pdf:pypdf",
            extracted_sha256=hashlib.sha256(extracted_json_bytes).hexdigest(),
            extracted_text_sha256=hashlib.sha256(b"current").hexdigest(),
        )

        assert resolved == stored_sha

    def test_returns_none_when_no_matching_record_exists(self, tmp_path: Path) -> None:
        resolved = stored_extraction_sha256(
            tmp_path,
            RAW_SHA,
            extractor="pdf:pypdf",
            extracted_sha256="d" * 64,
            extracted_text_sha256="e" * 64,
        )

        assert resolved is None


class TestCurrentExtractionRecords:
    """F3(b): a currentness query that never requires re-extraction."""

    def test_record_matching_todays_code_and_pypdf_identity_is_current(self, tmp_path: Path) -> None:
        identity = semantic_deps.extraction_identity()
        extraction_sha = _store(
            tmp_path,
            extractor_code_sha256=identity.code_sha256,
            pypdf_version=identity.pypdf_version,
            text="current",
        )

        current = current_extraction_records(tmp_path, RAW_SHA)

        assert {r.extraction_sha256 for r in current} == {extraction_sha}

    def test_record_with_stale_code_sha_is_not_current(self, tmp_path: Path) -> None:
        _store(tmp_path, extractor_code_sha256="0" * 64, text="stale")

        assert current_extraction_records(tmp_path, RAW_SHA) == []

    def test_stale_code_sha_alone_is_enough_to_be_non_current(self, tmp_path: Path) -> None:
        """The CODE-sha check must reject on its own, with pypdf pinned to today's.

        The test above leaves ``pypdf_version`` at the helper default, which is not
        the installed version, so the pypdf check rejects that record too. With two
        guards both rejecting, neither one is actually under test: deleting the
        code-sha comparison entirely leaves it green. This is the exact scenario the
        code-sha guard exists for -- Carmel's own extraction code changed while the
        third-party dependency did not, which is precisely the case
        ``_extractor_identity`` could never distinguish and this store was built to.
        """
        identity = semantic_deps.extraction_identity()
        _store(
            tmp_path,
            extractor_code_sha256="0" * 64,
            pypdf_version=identity.pypdf_version,
            text="stale code, current pypdf",
        )

        assert current_extraction_records(tmp_path, RAW_SHA) == []

    def test_no_records_returns_empty_list(self, tmp_path: Path) -> None:
        assert current_extraction_records(tmp_path, RAW_SHA) == []

    def test_never_requires_re_extraction_no_digest_arguments_accepted(self, tmp_path: Path) -> None:
        """Sanity check on the signature itself: only (workspace_root, raw_sha256)."""
        identity = semantic_deps.extraction_identity()
        _store(tmp_path, extractor_code_sha256=identity.code_sha256, pypdf_version=identity.pypdf_version)

        # No extracted_sha256/extracted_text_sha256 kwarg exists to pass -- this call
        # succeeding with only two positional args IS the assertion.
        current_extraction_records(tmp_path, RAW_SHA)


class TestPathAndShaValidation:
    def test_load_extraction_record_rejects_path_traversal_in_raw_sha(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_extraction_record(tmp_path, "../../../../etc/passwd", "a" * 64)

    def test_load_extraction_record_rejects_path_traversal_in_extraction_sha(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_extraction_record(tmp_path, "a" * 64, "../../../../etc/passwd")

    def test_load_extraction_record_rejects_malformed_raw_sha_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_extraction_record(tmp_path, "not-a-sha", "a" * 64)

    def test_load_extraction_record_rejects_malformed_extraction_sha_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_extraction_record(tmp_path, "a" * 64, "not-a-sha")

    def test_verify_extraction_record_rejects_bad_sha_shapes_too(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            verify_extraction_record(tmp_path, "../etc", "a" * 64)

    def test_list_extraction_records_rejects_bad_raw_sha_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            list_extraction_records(tmp_path, "../etc")


class TestLoadAndVerifyExtractionRecord:
    def test_round_trip_load_matches_stored_fields(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)

        record = load_extraction_record(tmp_path, RAW_SHA, extraction_sha)

        assert record is not None
        assert record.extraction_sha256 == extraction_sha
        assert record.parent_raw_sha256 == RAW_SHA
        assert record.extractor == "pdf:pypdf"

    def test_load_missing_record_returns_none(self, tmp_path: Path) -> None:
        assert load_extraction_record(tmp_path, RAW_SHA, "f" * 64) is None

    def test_verify_true_for_intact_record(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)

        assert verify_extraction_record(tmp_path, RAW_SHA, extraction_sha) is True

    def test_verify_false_when_extracted_json_corrupted_after_store(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        (dest / "extracted.json").write_bytes(b"corrupted")

        assert verify_extraction_record(tmp_path, RAW_SHA, extraction_sha) is False

    def test_text_txt_is_not_digest_checked(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        (dest / "text.txt").write_text("tampered, but never checked", encoding="utf-8")

        assert verify_extraction_record(tmp_path, RAW_SHA, extraction_sha) is True


class TestLoadAndVerifyDistinguishAbsentFromCorruptMeta:
    """F2/F8: absent meta.json -> None; corrupt/malformed meta.json -> ExtractionRecordError."""

    def test_load_returns_none_for_a_directory_with_no_meta_json_at_all(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        (dest / "meta.json").unlink()

        assert load_extraction_record(tmp_path, RAW_SHA, extraction_sha) is None

    def test_load_raises_for_meta_json_that_is_not_valid_json(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        (dest / "meta.json").write_text("not json {{{", encoding="utf-8")

        with pytest.raises(ExtractionRecordError):
            load_extraction_record(tmp_path, RAW_SHA, extraction_sha)

    def test_load_raises_for_meta_json_missing_a_required_field(self, tmp_path: Path) -> None:
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        raw = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        del raw["extractor_code_sha256"]
        (dest / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ExtractionRecordError):
            load_extraction_record(tmp_path, RAW_SHA, extraction_sha)

    def test_load_raises_for_meta_json_with_a_non_string_stored_at(self, tmp_path: Path) -> None:
        """Confirms the stored_at-sort-time TypeError bug is closed at LOAD time, not sort time."""
        extraction_sha = _store(tmp_path)
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        raw = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        raw["stored_at"] = 12345
        (dest / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        with pytest.raises(ExtractionRecordError):
            load_extraction_record(tmp_path, RAW_SHA, extraction_sha)

    def test_list_extraction_records_skips_a_record_with_non_string_stored_at_rather_than_crashing(
        self, tmp_path: Path
    ) -> None:
        """The original bug: sorted() on a mix of str/int stored_at raised an opaque TypeError.

        Now closed by load-time validation inside list_extraction_records' own
        try/except around _load_meta -- a malformed record is skipped (with a
        warning), and the OTHER, valid record is still returned; the whole call
        never raises.
        """
        good_sha = _store(tmp_path, text="good")
        bad_sha = _store(tmp_path, extractor_code_sha256="f" * 64, text="bad")
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / bad_sha
        raw = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        raw["stored_at"] = 12345
        (dest / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        records = list_extraction_records(tmp_path, RAW_SHA)

        assert {r.extraction_sha256 for r in records} == {good_sha}

    def test_load_returns_none_when_meta_json_was_moved_to_the_wrong_address_directory(self, tmp_path: Path) -> None:
        """F2: self-authentication -- a meta.json copied verbatim into a different
        address's directory must not verify there.
        """
        first_sha = _store(tmp_path, text="one")
        second_sha = _store(tmp_path, extractor_code_sha256="f" * 64, text="two")

        first_meta_path = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / first_sha / "meta.json"
        second_dir = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / second_sha
        (second_dir / "meta.json").write_bytes(first_meta_path.read_bytes())

        assert load_extraction_record(tmp_path, RAW_SHA, second_sha) is None

    def test_load_returns_none_when_identity_fields_do_not_recompute_to_the_address(self, tmp_path: Path) -> None:
        """F2: the address must be RECOMPUTED from the meta's own identity fields.

        The moved-meta test above is caught by the cheaper check that
        ``meta.extraction_sha256`` equals the directory name, so it never reaches
        the recomputation and does not test it. A forger who edits an identity field
        would of course also fix up ``extraction_sha256`` to match the directory --
        and then only recomputing the address from ``extractor_code_sha256`` and the
        other identity fields can tell that the record is lying about what produced
        it. That is the whole reason the address is content-derived rather than a
        directory naming convention, so it needs a test that fails when it is gone.
        """
        extraction_sha = _store(tmp_path, text="one")
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        raw = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        # Claim a different extractor code sha while leaving `extraction_sha256`
        # (and the directory name) untouched, so every OTHER check still passes.
        raw["extractor_code_sha256"] = "9" * 64
        assert raw["extraction_sha256"] == extraction_sha
        (dest / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        assert load_extraction_record(tmp_path, RAW_SHA, extraction_sha) is None
        assert verify_extraction_record(tmp_path, RAW_SHA, extraction_sha) is False

    def test_load_returns_none_when_meta_json_claims_a_different_parent_raw_sha(self, tmp_path: Path) -> None:
        """F2: self-authentication -- a meta.json whose own parent_raw_sha256
        disagrees with the caller's must not verify.
        """
        extraction_sha = _store(tmp_path, raw_sha256=RAW_SHA, text="one")
        dest = tmp_path / "evidence" / "literature" / RAW_SHA / "extractions" / extraction_sha
        raw = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
        raw["parent_raw_sha256"] = OTHER_RAW_SHA
        (dest / "meta.json").write_text(json.dumps(raw), encoding="utf-8")

        assert load_extraction_record(tmp_path, RAW_SHA, extraction_sha) is None
