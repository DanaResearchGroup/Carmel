"""Tests for carmel.services.evidence: content-addressed literature artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from carmel.agents.tools.extract import ExtractedText, TextSection
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.campaign import ReactorType
from carmel.schemas.literature import Citation, ExperimentalBenchmarkPayload, GroundingStatus, ObservableKind, Quantity
from carmel.services.evidence import (
    artifact_dir,
    load_artifact_text,
    store_artifact,
    verify_artifact,
)
from carmel.services.grounding import ground_finding

MAX_BYTES = 10_000_000


def _artifact(data: bytes, *, url: str = "https://example.org/paper.pdf", sha256: str | None = None) -> FetchedArtifact:
    digest = sha256 if sha256 is not None else hashlib.sha256(data).hexdigest()
    return FetchedArtifact(
        url=url,
        final_url=url,
        sha256=digest,
        content_type="application/pdf",
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )


def _extracted(text: str = "hello world") -> ExtractedText:
    return ExtractedText(text=text, normalized=text.casefold(), sections=[], extractor="pdf:pypdf", lossy=False)


class TestStoreArtifact:
    def test_creates_three_files_under_sha256_directory(self, tmp_path: Path) -> None:
        data = b"%PDF-1.4 hello"
        digest = hashlib.sha256(data).hexdigest()
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        dest = tmp_path / "evidence" / "literature" / digest
        assert dest == artifact_dir(tmp_path, digest)
        assert (dest / "raw.bin").read_bytes() == data
        assert (dest / "text.txt").read_text(encoding="utf-8") == "hello world"
        assert (dest / "meta.json").exists()
        assert stored.sha256 == digest

    def test_returned_metadata_round_trips_through_meta_json(self, tmp_path: Path) -> None:
        data = b"some raw bytes"
        artifact = _artifact(data, url="https://example.org/x")
        extracted = _extracted("extracted body")

        stored = store_artifact(
            tmp_path, data=data, artifact=artifact, extracted=extracted, license_note="CC-BY", max_bytes=MAX_BYTES
        )

        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        from carmel.services.artifacts import read_json

        reloaded = read_json(meta_path)
        assert reloaded["sha256"] == stored.sha256
        assert reloaded["source_url"] == stored.source_url == "https://example.org/x"
        assert reloaded["final_url"] == stored.final_url
        assert reloaded["content_type"] == stored.content_type
        assert reloaded["n_bytes"] == stored.n_bytes == len(data)
        assert reloaded["extractor"] == stored.extractor == "pdf:pypdf"
        assert reloaded["lossy"] == stored.lossy is False
        assert reloaded["license_note"] == stored.license_note == "CC-BY"

    def test_storing_same_bytes_twice_is_idempotent(self, tmp_path: Path) -> None:
        data = b"identical content"
        artifact = _artifact(data)
        extracted = _extracted()

        first = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        dest = artifact_dir(tmp_path, first.sha256)
        raw_path = dest / "raw.bin"
        mtime_before = raw_path.stat().st_mtime_ns

        literature_dir = tmp_path / "evidence" / "literature"
        assert len(list(literature_dir.iterdir())) == 1

        second = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert second == first
        assert raw_path.stat().st_mtime_ns == mtime_before
        assert len(list(literature_dir.iterdir())) == 1

    def test_repairs_missing_meta_json_when_raw_bin_exists(self, tmp_path: Path) -> None:
        data = b"needs repair"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        meta_path.unlink()

        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert meta_path.exists()
        assert repaired.sha256 == stored.sha256

    def test_sha256_mismatch_raises_value_error(self, tmp_path: Path) -> None:
        data = b"real bytes"
        wrong_artifact = _artifact(data, sha256="0" * 64)
        extracted = _extracted()

        with pytest.raises(ValueError, match="sha256 mismatch"):
            store_artifact(tmp_path, data=data, artifact=wrong_artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert not (tmp_path / "evidence").exists()

    def test_oversized_data_raises_value_error(self, tmp_path: Path) -> None:
        data = b"x" * 100
        artifact = _artifact(data)
        extracted = _extracted()

        with pytest.raises(ValueError, match="max_bytes"):
            store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=10)

        assert not (tmp_path / "evidence").exists()

    def test_empty_data_raises_value_error(self, tmp_path: Path) -> None:
        """A zero-byte artifact is never legitimate evidence: the store must refuse it
        outright, as defence in depth behind the acquisition layer's own empty-fetch
        check (the layer that lied about this in the live incident this guards
        against)."""
        data = b""
        artifact = _artifact(data)
        extracted = _extracted()

        with pytest.raises(ValueError, match="zero-byte"):
            store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert not (tmp_path / "evidence").exists()

    def test_hostile_source_url_path_traversal_does_not_affect_path(self, tmp_path: Path) -> None:
        data = b"hostile url payload"
        artifact = _artifact(data, url="http://x/../../../etc/passwd")
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        dest = artifact_dir(tmp_path, stored.sha256)
        assert dest.parent == tmp_path / "evidence" / "literature"
        assert dest.name == stored.sha256
        assert (dest / "raw.bin").exists()
        # nothing was written outside tmp_path
        for root, _dirs, files in os.walk(tmp_path):
            for name in files:
                assert Path(root, name).resolve().is_relative_to(tmp_path.resolve())

    def test_hostile_source_url_with_nul_and_newline_does_not_affect_path(self, tmp_path: Path) -> None:
        data = b"hostile url payload 2"
        hostile_url = "http://x/evil\x00\nname"
        artifact = _artifact(data, url=hostile_url)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        dest = artifact_dir(tmp_path, stored.sha256)
        assert dest.exists()
        assert stored.source_url == hostile_url
        for root, _dirs, files in os.walk(tmp_path):
            for name in files:
                assert Path(root, name).resolve().is_relative_to(tmp_path.resolve())

    def test_relative_workspace_root_resolves_inside_intended_workspace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "ws").mkdir()
        data = b"relative root bytes"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(Path("ws"), data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        dest = tmp_path / "ws" / "evidence" / "literature" / stored.sha256
        assert (dest / "raw.bin").exists()

    def test_symlinked_workspace_root_resolves_and_stays_inside_tmp_path(self, tmp_path: Path) -> None:
        real_workspace = tmp_path / "real_workspace"
        real_workspace.mkdir()
        link_workspace = tmp_path / "link_workspace"
        link_workspace.symlink_to(real_workspace, target_is_directory=True)

        data = b"symlinked root bytes"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(link_workspace, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        dest = real_workspace / "evidence" / "literature" / stored.sha256
        assert (dest / "raw.bin").read_bytes() == data
        for root, _dirs, files in os.walk(tmp_path):
            for name in files:
                assert Path(root, name).resolve().is_relative_to(tmp_path.resolve())


class TestLoadArtifactText:
    def test_returns_none_for_unknown_digest(self, tmp_path: Path) -> None:
        assert load_artifact_text(tmp_path, "0" * 64) is None

    def test_returns_extracted_text_for_stored_artifact(self, tmp_path: Path) -> None:
        data = b"loadable content"
        artifact = _artifact(data)
        extracted = _extracted("loadable content as text")

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        loaded = load_artifact_text(tmp_path, stored.sha256)
        assert loaded is not None
        assert loaded.text == "loadable content as text"
        assert loaded.extractor == "pdf:pypdf"
        assert loaded.lossy is False


class TestVerifyArtifact:
    def test_returns_true_for_intact_artifact(self, tmp_path: Path) -> None:
        data = b"intact bytes"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert verify_artifact(tmp_path, stored.sha256) is True

    def test_returns_false_for_unknown_digest(self, tmp_path: Path) -> None:
        assert verify_artifact(tmp_path, "0" * 64) is False

    def test_returns_false_after_raw_bytes_corrupted(self, tmp_path: Path) -> None:
        data = b"will be corrupted"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        raw_path.write_bytes(b"corrupted!!")

        assert verify_artifact(tmp_path, stored.sha256) is False


class TestTheExtractedSidecarIsVerifiedToo:
    """``raw.bin`` was content-addressed and re-hashed on every read; the sidecar was
    not checked at all -- and the sidecar is the file the grounding gate reads.

    Not a tamper defence: anyone who can write into the store can rewrite meta.json
    and the report beside it. It closes the non-adversarial case that actually
    happens -- a sidecar truncated by a full disk or an interrupted write, which can
    still parse as valid JSON while holding only part of the document.
    """

    def test_a_stored_artifact_records_its_sidecar_digest(self, tmp_path: Path) -> None:
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )

        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        assert stored.extracted_sha256 == hashlib.sha256(extracted_path.read_bytes()).hexdigest(), (
            "the recorded digest must describe the bytes actually written to disk"
        )

    def test_an_artifact_with_unreadable_meta_is_not_reported_as_verified(self, tmp_path: Path) -> None:
        """Copilot review. "Could not check" must not read as "intact".

        Without ``meta.json`` there is no recorded ``extracted_sha256``, so whether
        this artifact ever had a verifiable sidecar is unknown -- and the grounding
        gate quotes from ``extracted.json``. Returning True here reported a check that
        could not run as a check that passed: the same assertion-for-observation
        conflation already fixed twice on the acquisition side.
        """
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        assert verify_artifact(tmp_path, stored.sha256) is True

        (artifact_dir(tmp_path, stored.sha256) / "meta.json").write_text("{not json", encoding="utf-8")

        assert verify_artifact(tmp_path, stored.sha256) is False

    def test_a_legacy_artifact_with_no_recorded_sidecar_digest_still_verifies(self, tmp_path: Path) -> None:
        """Failing closed on unreadable meta must not fail closed on OLD meta.

        An artifact stored before ``extracted_sha256`` existed has READABLE metadata
        that simply records no sidecar digest. That is a known, benign state -- not an
        unknown one -- so it still verifies on ``raw.bin`` alone.
        """
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        del payload["extracted_sha256"]
        meta_path.write_text(json.dumps(payload), encoding="utf-8")

        assert verify_artifact(tmp_path, stored.sha256) is True

    def test_a_truncated_sidecar_that_still_parses_is_refused(self, tmp_path: Path) -> None:
        """The failure this exists for. Truncation that breaks JSON was already caught
        by the parse check; truncation that leaves VALID JSON was not, and that is the
        one that silently grounds quotes against a fragment of the document."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path,
            data=data,
            artifact=_artifact(data),
            extracted=_extracted("the full text of the paper, all of it"),
            max_bytes=MAX_BYTES,
        )
        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"

        # Still valid JSON, still a valid ExtractedText -- just not the whole document.
        payload = json.loads(extracted_path.read_text(encoding="utf-8"))
        payload["text"] = "the full text"
        extracted_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        assert ExtractedText.model_validate(json.loads(extracted_path.read_text(encoding="utf-8")))

        assert verify_artifact(tmp_path, stored.sha256) is False, (
            "a sidecar that parses but no longer matches its digest must not verify"
        )

    def test_raw_bytes_intact_is_not_enough_on_its_own(self, tmp_path: Path) -> None:
        """Guards against a fix that checks the sidecar only when raw.bin is damaged.
        raw.bin is untouched here; the sidecar alone diverges."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"

        extracted_path.write_text(
            json.dumps(
                {
                    "text": "different",
                    "normalized": "different",
                    "sections": [],
                    "extractor": "pdf:pypdf",
                    "lossy": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == stored.sha256, "raw.bin must be untouched"
        assert verify_artifact(tmp_path, stored.sha256) is False

    def test_an_artifact_stored_before_the_digest_existed_still_verifies(self, tmp_path: Path) -> None:
        """Backward compatibility, and the direction that matters. ``None`` means "this
        predates the field", NOT "corrupt". Refusing these would make every previously
        stored paper unreadable -- discarding good evidence to enforce a check that
        could not have run when it was written."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        del payload["extracted_sha256"]
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        assert verify_artifact(tmp_path, stored.sha256) is True


class TestDeepVerification:
    """``verify_artifact(..., deep=True)``: a derivation-binding check.

    Re-running the extractor is NOT guaranteed to reproduce byte-identical output
    (``pypdf`` is an unpinned dependency, ``pypdf>=5.0``, and different installed
    versions can extract different text from identical PDF bytes), so this cannot be
    a true re-derivation proof. It instead binds the extractor identity, the raw
    digest, and the sidecar digest into one value at store time, and the deep check
    re-derives that binding from ``meta.json``'s OWN recorded fields and compares it
    to the stored one. This is what catches a stale/swapped ``extracted.json`` whose
    ``extracted_sha256`` was updated to match the new (wrong) sidecar bytes but whose
    ``derivation_binding`` was left stale -- exactly the shape of a bad extraction
    that would otherwise sail through the cheap digest-only check.
    """

    def test_deep_verification_passes_for_a_freshly_stored_artifact(self, tmp_path: Path) -> None:
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )

        assert stored.derivation_binding is not None
        assert stored.extractor_version is not None
        assert verify_artifact(tmp_path, stored.sha256, deep=True) is True

    def test_deep_verification_rejects_a_stale_extraction_the_cheap_check_would_miss(self, tmp_path: Path) -> None:
        """The load-bearing case: a stale/swapped ``extracted.json`` whose
        ``extracted_sha256`` was updated to match the new (wrong) bytes, so the cheap
        digest-only check alone would still pass -- exactly what a bad or stale
        extraction of a numeric dataset table would look like on disk."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted("original text"), max_bytes=MAX_BYTES
        )
        extracted_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"

        # Swap in a valid-but-wrong extraction (e.g. from stale bytes, or a different
        # extractor run) and update ONLY `extracted_sha256` -- the field the cheap
        # check reads -- to match it, exactly as a naive re-extraction script would
        # that is unaware `derivation_binding` needs to be recomputed too.
        wrong = ExtractedText(
            text="a different, wrong extraction",
            normalized="a different, wrong extraction",
            sections=[],
            extractor="pdf:pypdf",
            lossy=False,
        )
        extracted_path.write_text(wrong.model_dump_json(indent=2), encoding="utf-8")
        new_extracted_digest = hashlib.sha256(extracted_path.read_bytes()).hexdigest()

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["extracted_sha256"] = new_extracted_digest
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # The cheap check only compares the sidecar's digest to meta's recorded
        # value, both of which now agree with each other -- so it is fooled.
        assert verify_artifact(tmp_path, stored.sha256) is True
        # The deep check re-derives the binding from meta's OWN extractor_version +
        # sha256 + (now-updated) extracted_sha256 and finds it disagrees with the
        # `derivation_binding` recorded at store time, which still describes the
        # original extraction.
        assert verify_artifact(tmp_path, stored.sha256, deep=True) is False

    def test_deep_verification_fails_closed_when_binding_is_unreadable(self, tmp_path: Path) -> None:
        """Same fail-closed pattern as an unreadable meta.json: "could not check" must
        not read as "intact"."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["derivation_binding"] = "0" * 64
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        assert verify_artifact(tmp_path, stored.sha256, deep=True) is False

    def test_legacy_artifact_without_a_derivation_binding_still_passes_the_cheap_check(self, tmp_path: Path) -> None:
        """Backward compatibility, and the direction that matters: an artifact stored
        before this field existed must not become unreadable under the DEFAULT
        (cheap) check. Deep verification is opt-in and is a strictly stronger
        standard old artifacts never had a chance to meet, so it is fine -- and
        documented here -- for deep verification to refuse them; the cheap check
        (the backward-compatible default) is what must keep working."""
        data = b"paper bytes"
        stored = store_artifact(
            tmp_path, data=data, artifact=_artifact(data), extracted=_extracted(), max_bytes=MAX_BYTES
        )
        meta_path = artifact_dir(tmp_path, stored.sha256) / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        del payload["extractor_version"]
        del payload["derivation_binding"]
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        assert verify_artifact(tmp_path, stored.sha256) is True
        # Deep verification has no binding to check against: fail closed rather than
        # silently declare a legacy artifact re-derivation-proof when it never was.
        assert verify_artifact(tmp_path, stored.sha256, deep=True) is False


class TestCorruptionDetectionAndRepair:
    """DEFECT 1 regression: the idempotent path must not trust a corrupted raw.bin forever."""

    def test_corrupted_raw_bin_is_repaired_when_bytes_are_in_hand(self, tmp_path: Path) -> None:
        data = b"original good bytes"
        artifact = _artifact(data)
        extracted = _extracted("original good bytes as text")

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        raw_path = artifact_dir(tmp_path, stored.sha256) / "raw.bin"
        # Corrupt the on-disk bytes directly (bit rot / partial write / tampering):
        # meta.json and the directory name still claim `stored.sha256`, but raw.bin
        # no longer hashes to it.
        raw_path.write_bytes(b"CORRUPTED ON DISK, DOES NOT MATCH DIGEST")
        assert verify_artifact(tmp_path, stored.sha256) is False

        # Re-storing with the same (correct) bytes/artifact/extracted must detect the
        # mismatch on the idempotent path and repair rather than silently trusting the
        # existing meta.json.
        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        # `stored_at` is legitimately refreshed by a repair (it is a genuine rewrite),
        # so compare on the fields that matter: the repaired artifact must describe the
        # bytes actually on disk, not merely equal the stale prior metadata object.
        assert repaired.sha256 == stored.sha256
        assert repaired.n_bytes == stored.n_bytes == len(data)
        assert raw_path.read_bytes() == data
        assert verify_artifact(tmp_path, stored.sha256) is True

    def test_meta_json_byte_count_disagreement_triggers_repair(self, tmp_path: Path) -> None:
        data = b"consistent bytes for this test"
        artifact = _artifact(data)
        extracted = _extracted()

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        dest = artifact_dir(tmp_path, stored.sha256)
        meta_path = dest / "meta.json"

        from carmel.services.artifacts import read_json, write_json

        tampered = read_json(meta_path)
        tampered["n_bytes"] = tampered["n_bytes"] + 1  # meta now disagrees with raw.bin's actual size
        write_json(meta_path, tampered)

        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert repaired.n_bytes == len(data)
        reloaded = read_json(meta_path)
        assert reloaded["n_bytes"] == len(data)

    def test_missing_text_txt_triggers_repair(self, tmp_path: Path) -> None:
        data = b"text file will be deleted"
        artifact = _artifact(data)
        extracted = _extracted("text file will be deleted as text")

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        text_path = artifact_dir(tmp_path, stored.sha256) / "text.txt"
        text_path.unlink()
        assert not text_path.exists()

        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert text_path.exists()
        assert repaired.sha256 == stored.sha256

    def test_missing_extracted_json_triggers_repair(self, tmp_path: Path) -> None:
        # spar round 5, Finding 2: the idempotent path re-stored identical bytes and
        # returned the existing meta.json WITHOUT checking that extracted.json (the
        # section-labelled source of truth the grounding gate depends on) still
        # exists -- silently treating a lost sidecar as intact forever, and leaving
        # future grounding with no structural evidence to check against.
        data = b"paper whose sidecar goes missing"
        artifact = _artifact(data)
        sections = [TextSection(label="body", start=0, end=5), TextSection(label="references", start=5, end=33)]
        extracted = ExtractedText(
            text="paper whose sidecar goes missing",
            normalized="paper whose sidecar goes missing",
            sections=sections,
            extractor="pdf:pypdf",
            lossy=False,
        )

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        extracted_json_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        extracted_json_path.unlink()
        assert not extracted_json_path.exists()

        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert extracted_json_path.exists()
        assert repaired.sha256 == stored.sha256
        loaded = load_artifact_text(tmp_path, stored.sha256)
        assert loaded is not None
        assert [s.label for s in loaded.sections] == ["body", "references"]
        assert loaded.lossy is False

    def test_truncated_unparseable_extracted_json_triggers_repair(self, tmp_path: Path) -> None:
        # A partial write (crash mid-write, disk full) can leave extracted.json present
        # but not valid JSON / not a valid ExtractedText -- this must be treated the
        # same as "missing", not trusted just because the file exists.
        data = b"paper whose sidecar gets truncated"
        artifact = _artifact(data)
        sections = [TextSection(label="body", start=0, end=5)]
        extracted = ExtractedText(
            text="paper whose sidecar gets truncated",
            normalized="paper whose sidecar gets truncated",
            sections=sections,
            extractor="pdf:pypdf",
            lossy=False,
        )

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        extracted_json_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        extracted_json_path.write_text("{not valid json", encoding="utf-8")

        repaired = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        assert repaired.sha256 == stored.sha256
        loaded = load_artifact_text(tmp_path, stored.sha256)
        assert loaded is not None
        assert [s.label for s in loaded.sections] == ["body"]
        assert loaded.lossy is False


class TestExtractedJsonPersistsSections:
    """DEFECT 2 regression: reload must retain section labels the grounding gate needs."""

    def test_reloaded_artifact_creates_extracted_json(self, tmp_path: Path) -> None:
        data = b"paper bytes"
        artifact = _artifact(data)
        sections = [TextSection(label="body", start=0, end=5), TextSection(label="references", start=5, end=11)]
        extracted = ExtractedText(
            text="hello world",
            normalized="hello world",
            sections=sections,
            extractor="pdf:pypdf",
            lossy=False,
        )

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)

        extracted_json_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        assert extracted_json_path.exists()

        loaded = load_artifact_text(tmp_path, stored.sha256)
        assert loaded is not None
        assert loaded.lossy is False
        assert [s.label for s in loaded.sections] == ["body", "references"]

    def test_reloaded_artifact_retains_references_section_end_to_end_through_ground_finding(
        self, tmp_path: Path
    ) -> None:
        """The whole point of persisting sections: a re-grounded quote from the
        references section must still be rejected as REFERENCES_ONLY after a reload,
        not silently accepted because sections came back empty."""
        body = "Body content discussing other matters entirely, unrelated to the quoted material below.\n\n"
        references_heading = "References\n"
        reference_entry = (
            "Smith, J. (2019). The measured ignition delay time at 1200 K was 850 microseconds. Journal of Combustion."
        )
        text = body + references_heading + reference_entry
        references_start = len(body)
        sections = [
            TextSection(label="body", start=0, end=references_start),
            TextSection(label="references", start=references_start, end=len(text)),
        ]
        extracted = ExtractedText(
            text=text,
            normalized=text.casefold(),
            sections=sections,
            extractor="pdf:pypdf",
            lossy=False,
        )
        data = b"raw bytes for the references-section paper"
        artifact = _artifact(data)

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        reloaded = load_artifact_text(tmp_path, stored.sha256)
        assert reloaded is not None

        quote = "The measured ignition delay time at 1200 K was 850 microseconds"
        citation = Citation(title="Ignition delay times study", authors=["Smith, J."], year=2019, doi="10.1000/xyz123")
        payload = ExperimentalBenchmarkPayload(
            reactor_type=ReactorType.SHOCK_TUBE,
            observable=ObservableKind.IGNITION_DELAY_TIME,
            observable_raw="ignition delay time",
            measured=[Quantity(value=850.0, unit="microseconds")],
        )

        verdict = ground_finding(payload=payload, citation=citation, quote=quote, extracted=reloaded)

        assert verdict.status == GroundingStatus.REFERENCES_ONLY
        assert verdict.grounded is False

    def test_old_layout_without_extracted_json_reloads_as_degraded(self, tmp_path: Path) -> None:
        data = b"old layout bytes"
        artifact = _artifact(data)
        sections = [TextSection(label="body", start=0, end=5), TextSection(label="references", start=5, end=16)]
        extracted = ExtractedText(
            text="old layout bytes as text",
            normalized="old layout bytes as text",
            sections=sections,
            extractor="pdf:pypdf",
            lossy=False,
        )

        stored = store_artifact(tmp_path, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)
        extracted_json_path = artifact_dir(tmp_path, stored.sha256) / "extracted.json"
        assert extracted_json_path.exists()
        extracted_json_path.unlink()  # simulate an artifact stored by the old (pre-fix) layout

        loaded = load_artifact_text(tmp_path, stored.sha256)

        assert loaded is not None
        assert loaded.sections == []
        assert loaded.lossy is True  # forced True regardless of the originally stored value
        assert loaded.text == "old layout bytes as text"


class TestSha256Validation:
    """DEFECT 3 regression: read paths must validate sha256 like the write path does."""

    def test_load_artifact_text_rejects_path_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid sha256"):
            load_artifact_text(tmp_path, "../../etc/passwd")

    def test_verify_artifact_rejects_path_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="invalid sha256"):
            verify_artifact(tmp_path, "../../etc/passwd")

    def test_load_artifact_text_rejects_63_char_hex(self, tmp_path: Path) -> None:
        short_digest = "0" * 63
        with pytest.raises(ValueError, match="invalid sha256"):
            load_artifact_text(tmp_path, short_digest)

    def test_verify_artifact_rejects_63_char_hex(self, tmp_path: Path) -> None:
        short_digest = "0" * 63
        with pytest.raises(ValueError, match="invalid sha256"):
            verify_artifact(tmp_path, short_digest)
