"""Tests for carmel.services.evidence: content-addressed literature artifact storage."""

from __future__ import annotations

import hashlib
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
